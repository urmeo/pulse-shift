const FEATURE_NAMES = [
  "heat_index_f", "cold_stress", "heat_stress", "aqi", "humidity", "wind_mph",
  "precip_in", "visibility_mi", "smoke_haze", "hour_sin", "hour_cos", "is_weekend",
];
const NUMERIC_INPUTS = ["temp", "humidity", "aqi", "wind", "precip", "visibility"];

export function heatIndex(temperatureF, humidity) {
  const simple = (0.5 * (temperatureF + 61 + (temperatureF - 68) * 1.2 + humidity * 0.094) + temperatureF) / 2;
  let heatIndexF = simple;
  if (simple >= 80) {
    heatIndexF = -42.379 + 2.04901523 * temperatureF + 10.14333127 * humidity -
      0.22475541 * temperatureF * humidity - 0.00683783 * temperatureF * temperatureF -
      0.05481717 * humidity * humidity + 0.00122874 * temperatureF * temperatureF * humidity +
      0.00085282 * temperatureF * humidity * humidity -
      0.00000199 * temperatureF * temperatureF * humidity * humidity;
    if (humidity < 13 && temperatureF >= 80 && temperatureF <= 112) {
      heatIndexF -= ((13 - humidity) / 4) * Math.sqrt((17 - Math.abs(temperatureF - 95)) / 17);
    } else if (humidity > 85 && temperatureF >= 80 && temperatureF <= 87) {
      heatIndexF += ((humidity - 85) / 10) * ((87 - temperatureF) / 5);
    }
  }
  const scaled = heatIndexF * 10;
  const floor = Math.floor(scaled);
  // NumPy rounding.
  const rounded = scaled - floor === 0.5 ? floor + (Math.abs(floor) % 2) : Math.round(scaled);
  return rounded / 10;
}

export class DemandModel {
  #parameters;
  #intercept;
  #stress;

  constructor(artifact) {
    const names = artifact?.features;
    const validFeatures = Array.isArray(names) && names.length === FEATURE_NAMES.length &&
      new Set(names).size === FEATURE_NAMES.length && FEATURE_NAMES.every((name) => names.includes(name));
    const validParameters = ["mean", "scale", "coef"].every((field) =>
      Array.isArray(artifact?.[field]) && artifact[field].length === FEATURE_NAMES.length &&
      FEATURE_NAMES.every((_, index) => Number.isFinite(artifact[field][index]))
    );
    if (!validFeatures || !validParameters || !artifact.scale.every((scale) => scale > 0) ||
        !Number.isFinite(artifact.intercept) ||
        !Number.isFinite(artifact.stress?.cold_base_f) || !Number.isFinite(artifact.stress?.heat_base_f) ||
        !Number.isFinite(artifact.safety?.heat_unsafe_f) || !Number.isFinite(artifact.safety?.aqi_unsafe)) {
      throw new Error("Model artifact is invalid.");
    }
    this.#parameters = names.map((name, index) => ({
      name, mean: artifact.mean[index], scale: artifact.scale[index], coefficient: artifact.coef[index],
    }));
    this.#intercept = artifact.intercept;
    this.#stress = { ...artifact.stress };
    this.meta = Object.freeze({ ...artifact.meta });
    this.safety = Object.freeze({ ...artifact.safety });
    Object.freeze(this);
  }

  features(input) {
    if (!input || !NUMERIC_INPUTS.every((name) => Number.isFinite(input[name])) ||
        !Number.isInteger(input.hour) || input.hour < 0 || input.hour > 23 ||
        typeof input.weekend !== "boolean" || typeof input.smoke !== "boolean") {
      throw new Error("Model inputs are invalid.");
    }
    const angle = (2 * Math.PI * input.hour) / 24;
    const heatIndexF = heatIndex(input.temp, input.humidity);
    const values = {
      heat_index_f: heatIndexF,
      cold_stress: Math.max(0, this.#stress.cold_base_f - input.temp),
      heat_stress: Math.max(0, heatIndexF - this.#stress.heat_base_f),
      aqi: input.aqi,
      humidity: input.humidity,
      wind_mph: input.wind,
      precip_in: input.precip,
      visibility_mi: input.visibility,
      smoke_haze: input.smoke ? 1 : 0,
      hour_sin: Math.sin(angle),
      hour_cos: Math.cos(angle),
      is_weekend: input.weekend ? 1 : 0,
    };
    if (!Object.values(values).every(Number.isFinite)) throw new Error("Model inputs are invalid.");
    return values;
  }

  evaluate(input) {
    const values = this.features(input);
    let score = this.#intercept;
    for (const { name, mean, scale, coefficient } of this.#parameters) {
      score += coefficient * ((values[name] - mean) / scale);
    }
    if (!Number.isFinite(score)) throw new Error("Model score is invalid.");
    return {
      probability: 1 / (1 + Math.exp(-score)),
      heatIndexF: values.heat_index_f,
      excluded: values.heat_index_f >= this.safety.heat_unsafe_f || input.aqi >= this.safety.aqi_unsafe,
    };
  }

  selectBestHour(rows) {
    if (!Array.isArray(rows)) throw new Error("Forecast hours are invalid.");
    let best = null;
    for (const input of rows) {
      if (!Number.isSafeInteger(input?.epoch)) throw new Error("Forecast timestamp is invalid.");
      const estimate = this.evaluate(input);
      if (input.hour < 6 || input.hour > 21 || estimate.excluded) continue;
      if (!best || estimate.probability < best.probability ||
          (estimate.probability === best.probability && input.epoch < best.epoch)) {
        best = { ...input, ...estimate };
      }
    }
    return best;
  }
}
