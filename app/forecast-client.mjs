export const DC_TIME_ZONE = "America/New_York";
export const SECONDS_PER_HOUR = 3600;
export const MILLISECONDS_PER_HOUR = SECONDS_PER_HOUR * 1000;

const dcDateFormatter = new Intl.DateTimeFormat("en-US", {
  timeZone: DC_TIME_ZONE,
  hour: "numeric",
  hourCycle: "h23",
  weekday: "short",
});
const forecastQuery = "latitude=38.8951&longitude=-77.0364&forecast_hours=25&timeformat=unixtime&timezone=America%2FNew_York";
const weatherUrl = "https://api.open-meteo.com/v1/forecast?" + forecastQuery +
  "&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation,visibility&temperature_unit=fahrenheit&wind_speed_unit=mph&precipitation_unit=inch";
const airQualityUrl = "https://air-quality-api.open-meteo.com/v1/air-quality?" + forecastQuery + "&hourly=us_aqi";
const weatherFields = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation", "visibility"];

export function dcHourWeekend(epochSeconds) {
  const parts = dcDateFormatter.formatToParts(new Date(epochSeconds * 1000));
  const hour = Number(parts.find((part) => part.type === "hour").value);
  const weekday = parts.find((part) => part.type === "weekday").value;
  return { hour, weekend: weekday === "Sat" || weekday === "Sun" };
}

export function forecastInputs(weather, air, smoke, now = Date.now()) {
  const weatherUnits = weather?.hourly_units;
  const weatherHours = weather?.hourly;
  const airUnits = air?.hourly_units;
  const airHours = air?.hourly;
  if (!weatherUnits || weatherUnits.time !== "unixtime" || weatherUnits.temperature_2m !== "°F" ||
      weatherUnits.relative_humidity_2m !== "%" || !["mp/h", "mph"].includes(weatherUnits.wind_speed_10m) ||
      weatherUnits.precipitation !== "inch" || !["m", "ft"].includes(weatherUnits.visibility) ||
      !airUnits || airUnits.time !== "unixtime" || airUnits.us_aqi !== "USAQI" || !weatherHours || !airHours) {
    throw new Error("Unexpected forecast units or fields.");
  }
  if (!Array.isArray(weatherHours.time) ||
      !weatherFields.every((field) => Array.isArray(weatherHours[field]) && weatherHours[field].length === weatherHours.time.length) ||
      !Array.isArray(airHours.time) || !Array.isArray(airHours.us_aqi) || airHours.time.length !== airHours.us_aqi.length) {
    throw new Error("Incomplete forecast response.");
  }
  if (new Set(weatherHours.time).size !== weatherHours.time.length || new Set(airHours.time).size !== airHours.time.length) {
    throw new Error("Ambiguous forecast response: duplicate timestamps.");
  }
  const aqiByTime = new Map(airHours.time.map((epoch, index) => [epoch, airHours.us_aqi[index]]));
  const startEpoch = Math.floor(now / MILLISECONDS_PER_HOUR) * SECONDS_PER_HOUR;
  const visibilityUnitsPerMile = weatherUnits.visibility === "ft" ? 5280 : 1609.344;
  const rows = [];
  for (const [index, epoch] of weatherHours.time.entries()) {
    if (!Number.isSafeInteger(epoch) || epoch < startEpoch || epoch >= startEpoch + 24 * SECONDS_PER_HOUR) continue;
    const input = {
      epoch,
      temp: weatherHours.temperature_2m[index],
      humidity: weatherHours.relative_humidity_2m[index],
      wind: weatherHours.wind_speed_10m[index],
      precip: weatherHours.precipitation[index],
      aqi: aqiByTime.get(epoch),
      visibility: Math.min(10, weatherHours.visibility[index] / visibilityUnitsPerMile),
      ...dcHourWeekend(epoch),
      smoke,
    };
    if (!weatherFields.every((field) => Number.isFinite(weatherHours[field][index])) || !Number.isFinite(input.aqi) ||
        input.temp < -40 || input.temp > 130 || input.humidity < 0 || input.humidity > 100 ||
        input.aqi < 0 || input.aqi > 500 || input.wind < 0 || input.wind > 100 ||
        input.precip < 0 || input.precip > 5 || input.visibility < 0) continue;
    rows.push(input);
  }
  return rows.sort((first, second) => first.epoch - second.epoch);
}

export class ForecastClient {
  #fetcher;
  #timeoutMs;

  constructor({ fetcher = globalThis.fetch, timeoutMs = 8000 } = {}) {
    if (typeof fetcher !== "function") throw new TypeError("A forecast fetch function is required.");
    if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) throw new RangeError("Forecast timeout must be positive.");
    this.#fetcher = fetcher;
    this.#timeoutMs = timeoutMs;
  }

  async load() {
    const controller = new AbortController();
    const timeoutError = new Error("Forecast request timed out.");
    const timer = setTimeout(() => controller.abort(timeoutError), this.#timeoutMs);
    try {
      const [weather, air] = await Promise.all([
        this.#fetchJson(weatherUrl, controller.signal),
        this.#fetchJson(airQualityUrl, controller.signal),
      ]);
      return { weather, air };
    } catch (error) {
      if (controller.signal.reason === timeoutError) throw timeoutError;
      throw error;
    } finally {
      clearTimeout(timer);
      controller.abort();
    }
  }

  async #fetchJson(url, signal) {
    const fetcher = this.#fetcher;
    const response = await fetcher(url, { signal });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
  }
}
