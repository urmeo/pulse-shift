const model = window.PULSESHIFT_MODEL;
const element = (id) => document.getElementById(id);
const DC_TIME_ZONE = "America/New_York";
const COLD_BASE_F = model.stress.cold_base_f;
const HEAT_BASE_F = model.stress.heat_base_f;
let editRevision = 0;

function heatIndex(t, rh) {
  const simple = (0.5 * (t + 61 + (t - 68) * 1.2 + rh * 0.094) + t) / 2;
  let hi = simple;
  if (simple >= 80) {
    hi = -42.379 + 2.04901523 * t + 10.14333127 * rh - 0.22475541 * t * rh -
      0.00683783 * t * t - 0.05481717 * rh * rh + 0.00122874 * t * t * rh +
      0.00085282 * t * rh * rh - 0.00000199 * t * t * rh * rh;
    if (rh < 13 && t >= 80 && t <= 112) {
      hi -= ((13 - rh) / 4) * Math.sqrt((17 - Math.abs(t - 95)) / 17);
    } else if (rh > 85 && t >= 80 && t <= 87) {
      hi += ((rh - 85) / 10) * ((87 - t) / 5);
    }
  }
  const scaled = hi * 10;
  const floor = Math.floor(scaled);
  // Match NumPy's rounding of exact halves to the nearest even digit.
  const rounded = scaled - floor === 0.5 ? floor + (Math.abs(floor) % 2) : Math.round(scaled);
  return rounded / 10;
}

function features(input) {
  const angle = (2 * Math.PI * input.hour) / 24;
  const hi = heatIndex(input.temp, input.humidity);
  return {
    heat_index_f: hi,
    cold_stress: Math.max(0, COLD_BASE_F - input.temp),
    heat_stress: Math.max(0, hi - HEAT_BASE_F),
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
}

function predictSuppression(input) {
  const values = features(input);
  let score = model.intercept;
  model.features.forEach((name, i) => {
    if (!Number.isFinite(values[name]) || !Number.isFinite(model.coef[i]) || !(model.scale[i] > 0)) {
      throw new Error("Model inputs or coefficients are invalid.");
    }
    score += model.coef[i] * ((values[name] - model.mean[i]) / model.scale[i]);
  });
  if (!Number.isFinite(score)) throw new Error("Model score is invalid.");
  return 1 / (1 + Math.exp(-score));
}

function percent(p) {
  if (p < 0.01) return "<1%";
  if (p > 0.99) return ">99%";
  return Math.round(p * 100) + "%";
}

function riskClass(probability) {
  if (probability < 0.15) return "low";
  if (probability < 0.35) return "moderate";
  if (probability < 0.6) return "high";
  return "severe";
}

function readNumber(id) {
  const input = element(id);
  const value = input.valueAsNumber;
  const valid = Number.isFinite(value) && input.checkValidity();
  input.setAttribute("aria-invalid", String(!valid));
  if (!valid) {
    const label = input.closest("label").firstChild.textContent.trim();
    throw new Error(`Enter a valid value for ${label}.`);
  }
  return value;
}

function readInputs() {
  const hour = Number(element("hour").value);
  if (!Number.isInteger(hour) || hour < 0 || hour > 23) throw new Error("Select a start hour.");
  return {
    temp: readNumber("temp"), humidity: readNumber("humidity"), aqi: readNumber("aqi"),
    wind: readNumber("wind"), precip: readNumber("precip"), visibility: readNumber("visibility"),
    hour, weekend: element("weekend").checked, smoke: element("smoke").checked,
  };
}

function update() {
  element("besthour").textContent = "";
  try {
    const input = readInputs();
    const heatIndexF = heatIndex(input.temp, input.humidity);
    const probability = predictSuppression(input);
    const excluded = heatIndexF >= model.safety.heat_unsafe_f || input.aqi >= model.safety.aqi_unsafe;
    element("result").hidden = false;
    element("error").textContent = "";
    element("risk").className = "risk " + (excluded ? "severe" : riskClass(probability));
    element("pct").textContent = percent(probability);
    element("band").textContent = excluded ? "Study threshold exceeded" : "chance of low bike-share demand";
    element("reco").textContent = excluded
      ? `Forecast suggestions require a heat index below ${model.safety.heat_unsafe_f}°F and AQI below ${model.safety.aqi_unsafe}. Follow official advisories.`
      : "Estimated chance that network rides fall below half the expected count for this season, day type and hour.";
    element("detail").textContent = `Heat index ${heatIndexF}°F · AQI ${Math.round(input.aqi * 10) / 10}`;
  } catch (error) {
    element("result").hidden = true;
    element("error").textContent = error.message;
  }
}

function dcHourWeekend(epoch) {
  const parts = new Intl.DateTimeFormat("en-US", {
    timeZone: DC_TIME_ZONE, hour: "numeric", hourCycle: "h23", weekday: "short",
  }).formatToParts(new Date(epoch * 1000));
  const hour = Number(parts.find((x) => x.type === "hour").value);
  const weekday = parts.find((x) => x.type === "weekday").value;
  return { hour, weekend: weekday === "Sat" || weekday === "Sun" };
}

function forecastInputs(weather, air, smoke, now = Date.now()) {
  const weatherUnits = weather.hourly_units;
  const weatherHours = weather.hourly;
  const airHours = air.hourly;
  if (!weatherUnits || weatherUnits.time !== "unixtime" || weatherUnits.temperature_2m !== "°F" ||
      weatherUnits.relative_humidity_2m !== "%" || !["mp/h", "mph"].includes(weatherUnits.wind_speed_10m) ||
      weatherUnits.precipitation !== "inch" || !["m", "ft"].includes(weatherUnits.visibility) ||
      !air.hourly_units || air.hourly_units.time !== "unixtime" || air.hourly_units.us_aqi !== "USAQI" || !weatherHours || !airHours) {
    throw new Error("Unexpected forecast units or fields.");
  }
  const fields = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation", "visibility"];
  if (!Array.isArray(weatherHours.time) || !fields.every((f) => Array.isArray(weatherHours[f]) && weatherHours[f].length === weatherHours.time.length) ||
      !Array.isArray(airHours.time) || !Array.isArray(airHours.us_aqi) || airHours.time.length !== airHours.us_aqi.length) {
    throw new Error("Incomplete forecast response.");
  }
  if (new Set(weatherHours.time).size !== weatherHours.time.length || new Set(airHours.time).size !== airHours.time.length) {
    throw new Error("Ambiguous forecast response: duplicate timestamps.");
  }
  const aqiByTime = new Map(airHours.time.map((t, i) => [t, airHours.us_aqi[i]]));
  const start = Math.floor(now / 3600000) * 3600;
  const rows = [];
  weatherHours.time.forEach((epoch, i) => {
    if (!Number.isFinite(epoch) || epoch < start || epoch >= start + 86400) return;
    const input = {
      epoch, temp: weatherHours.temperature_2m[i], humidity: weatherHours.relative_humidity_2m[i],
      wind: weatherHours.wind_speed_10m[i], precip: weatherHours.precipitation[i], aqi: aqiByTime.get(epoch),
      visibility: Math.min(10, weatherHours.visibility[i] / (weatherUnits.visibility === "ft" ? 5280 : 1609.344)),
      ...dcHourWeekend(epoch), smoke,
    };
    if (!fields.every((f) => Number.isFinite(weatherHours[f][i])) || !Number.isFinite(input.aqi) ||
        input.temp < -40 || input.temp > 130 || input.humidity < 0 || input.humidity > 100 ||
        input.aqi < 0 || input.aqi > 500 || input.wind < 0 || input.wind > 100 ||
        input.precip < 0 || input.precip > 5 || input.visibility < 0) return;
    rows.push(input);
  });
  return rows.sort((x, y) => x.epoch - y.epoch);
}

function bestForecastHour(rows) {
  let best = null;
  for (const input of rows) {
    if (input.hour < 6 || input.hour > 21 || input.aqi >= model.safety.aqi_unsafe ||
        heatIndex(input.temp, input.humidity) >= model.safety.heat_unsafe_f) continue;
    const probability = predictSuppression(input);
    if (!best || probability < best.risk) best = { ...input, risk: probability };
  }
  return best;
}

async function fetchJson(url, ms = 8000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), ms);
  try {
    const response = await fetch(url, { signal: controller.signal });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return await response.json();
  } finally {
    clearTimeout(timer);
  }
}

async function liveWeather() {
  const button = element("live");
  if (button.disabled) return;
  const revision = editRevision;
  const now = Date.now();
  button.disabled = true;
  button.textContent = "Loading…";
  button.setAttribute("aria-busy", "true");
  element("error").textContent = "";
  element("besthour").textContent = "";
  element("forecastnote").textContent = "";
  try {
    const query = "latitude=38.8951&longitude=-77.0364&forecast_hours=25&timeformat=unixtime&timezone=America%2FNew_York";
    const [weather, air] = await Promise.all([
      fetchJson("https://api.open-meteo.com/v1/forecast?" + query +
        "&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation,visibility&temperature_unit=fahrenheit&wind_speed_unit=mph&precipitation_unit=inch"),
      fetchJson("https://air-quality-api.open-meteo.com/v1/air-quality?" + query + "&hourly=us_aqi"),
    ]);
    if (revision !== editRevision) {
      element("error").textContent = "Inputs changed during loading. Load the forecast again to replace them.";
      return;
    }
    const rows = forecastInputs(weather, air, element("smoke").checked, now);
    const first = rows.find((r) => r.epoch === Math.floor(now / 3600000) * 3600);
    if (!first) throw new Error("Current-hour weather or AQI is missing.");
    ["temp", "humidity", "aqi", "wind", "precip", "visibility", "hour"].forEach((id) => { element(id).value = first[id]; });
    element("weekend").checked = first.weekend;
    update();
    const best = bestForecastHour(rows);
    const stamp = best && new Intl.DateTimeFormat("en-US", {
      timeZone: DC_TIME_ZONE, weekday: "short", month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", hourCycle: "h23", timeZoneName: "short",
    }).format(new Date(best.epoch * 1000));
    element("besthour").textContent = best
      ? `Lowest estimated chance of low demand in the next 24 hours, within study limits: ${stamp} (${percent(best.risk)}, AQI ${Math.round(best.aqi)}).`
      : "No complete daytime forecast hour meets the study's heat and AQI limits.";
    element("forecastnote").textContent = "Open-Meteo weather and CAMS AQI forecasts. Hours from 06:00 to 21:00 DC time are compared. Smoke/haze uses your checkbox setting. Hours with missing data are excluded. Visibility is capped at 10 mi to match the training data.";
    button.textContent = "DC forecast loaded";
  } catch (error) {
    element("besthour").textContent = "";
    element("forecastnote").textContent = "";
    element("error").textContent = "Forecast unavailable: " + error.message + " Enter conditions manually.";
  } finally {
    if (button.textContent === "Loading…") button.textContent = "Load DC forecast";
    button.disabled = false;
    button.removeAttribute("aria-busy");
  }
}

function init() {
  const hourSelect = element("hour");
  for (let hour = 0; hour < 24; hour++) {
    const option = document.createElement("option");
    option.value = hour;
    option.textContent = String(hour).padStart(2, "0") + ":00";
    hourSelect.appendChild(option);
  }
  hourSelect.value = 17;
  document.querySelectorAll("input, select").forEach((el) => el.addEventListener("input", () => {
    editRevision++;
    if (!element("live").disabled) element("live").textContent = "Load DC forecast";
    element("forecastnote").textContent = "";
    update();
  }));
  element("form").addEventListener("submit", (event) => { event.preventDefault(); update(); });
  element("live").addEventListener("click", liveWeather);
  element("meta").textContent = `DC 2022–2024 · 2024 test AUROC ${model.meta.auroc_2024}`;
  element("meta").title = model.meta.metrics_note;
  element("modelnote").textContent = "This model estimates DC-area bike-share network demand, not personal exercise risk. The 2024 test used a model trained on 2022–2023; the app's model uses all three years. Live forecasts have not been prospectively validated.";
  update();
}

init();
