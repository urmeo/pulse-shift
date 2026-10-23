import {
  DC_TIME_ZONE, MILLISECONDS_PER_HOUR, SECONDS_PER_HOUR, forecastInputs,
} from "./forecast-client.mjs";

const CONDITION_LABELS = {
  temp: "Temperature (°F)",
  humidity: "Humidity (%)",
  aqi: "US Air Quality Index (AQI)",
  wind: "Wind (mph)",
  precip: "Precipitation (in/hr)",
  visibility: "Visibility (mi)",
};
const CONDITION_IDS = Object.keys(CONDITION_LABELS);
const INPUT_IDS = [...CONDITION_IDS, "hour", "weekend", "smoke"];
const FORECAST_TIME = new Intl.DateTimeFormat("en-US", {
  timeZone: DC_TIME_ZONE,
  weekday: "short", month: "short", day: "numeric", hour: "2-digit", minute: "2-digit",
  hourCycle: "h23", timeZoneName: "short",
});

export function percent(probability) {
  if (probability < 0.01) return "<1%";
  if (probability > 0.99) return ">99%";
  return Math.round(probability * 100) + "%";
}

function riskClass(probability) {
  if (probability < 0.15) return "low";
  if (probability < 0.35) return "moderate";
  if (probability < 0.6) return "high";
  return "severe";
}

export class PulseShiftApp {
  #document;
  #elements;
  #model;
  #forecasts;
  #now;
  #editRevision = 0;

  constructor(document, { model, forecastClient, now = Date.now }) {
    this.#document = document;
    this.#model = model;
    this.#forecasts = forecastClient;
    this.#now = now;
    const ids = [
      ...INPUT_IDS, "form", "live", "error", "result",
      "risk", "pct", "band", "reco", "detail", "besthour", "forecastnote", "meta", "modelnote",
    ];
    this.#elements = Object.fromEntries(ids.map((id) => [id, document.getElementById(id)]));
  }

  start() {
    const fields = this.#elements;
    for (let hour = 0; hour < 24; hour++) {
      const option = this.#document.createElement("option");
      option.value = hour;
      option.textContent = String(hour).padStart(2, "0") + ":00";
      fields.hour.appendChild(option);
    }
    fields.hour.value = 17;
    for (const id of INPUT_IDS) {
      fields[id].addEventListener("input", () => {
        this.#editRevision++;
        if (!fields.live.disabled) fields.live.textContent = "Load DC forecast";
        this.#clearForecast();
        this.render();
      });
    }
    fields.form.addEventListener("submit", (event) => {
      event.preventDefault();
      this.render();
    });
    fields.live.addEventListener("click", () => this.loadForecast());
    fields.meta.textContent = `DC 2022–2024 · 2024 test AUROC ${this.#model.meta.auroc_2024}`;
    fields.meta.title = this.#model.meta.metrics_note;
    fields.modelnote.textContent = "This model estimates DC-area bike-share network demand, not personal exercise risk. The 2024 test used a model trained on 2022–2023; the app's model uses all three years. Live forecasts have not been prospectively validated.";
    this.render();
  }

  readInputs() {
    const fields = this.#elements;
    const hour = String(fields.hour.value).trim() === "" ? NaN : Number(fields.hour.value);
    const hourValid = Number.isInteger(hour) && hour >= 0 && hour <= 23;
    fields.hour.setAttribute("aria-invalid", String(!hourValid));
    const values = { hour, weekend: fields.weekend.checked, smoke: fields.smoke.checked };
    let invalidLabel = null;
    for (const id of CONDITION_IDS) {
      const field = fields[id];
      const value = field.valueAsNumber;
      const valid = Number.isFinite(value) && field.checkValidity();
      field.setAttribute("aria-invalid", String(!valid));
      values[id] = value;
      if (!valid && !invalidLabel) invalidLabel = CONDITION_LABELS[id];
    }
    if (invalidLabel) throw new Error(`Enter a valid value for ${invalidLabel}.`);
    if (!hourValid) throw new Error("Select a start hour.");
    return values;
  }

  render() {
    const fields = this.#elements;
    try {
      const input = this.readInputs();
      const { probability, heatIndexF, excluded } = this.#model.evaluate(input);
      fields.result.hidden = false;
      fields.error.textContent = "";
      fields.risk.className = "risk " + (excluded ? "severe" : riskClass(probability));
      fields.pct.textContent = percent(probability);
      fields.band.textContent = excluded ? "Study threshold exceeded" : "chance of low bike-share demand";
      fields.reco.textContent = excluded
        ? `Forecast suggestions require a heat index below ${this.#model.safety.heat_unsafe_f}°F and AQI below ${this.#model.safety.aqi_unsafe}. Follow official advisories.`
        : "Estimated chance that network rides fall below half the expected count for this season, day type and hour.";
      fields.detail.textContent = `Heat index ${heatIndexF}°F · AQI ${Math.round(input.aqi * 10) / 10}`;
    } catch (error) {
      fields.result.hidden = true;
      fields.error.textContent = error.message;
    }
  }

  #clearForecast() {
    this.#elements.besthour.textContent = "";
    this.#elements.forecastnote.textContent = "";
  }

  #isStale(revision) {
    if (revision === this.#editRevision) return false;
    if (!this.#elements.result.hidden) {
      this.#elements.error.textContent = "Inputs changed during loading. Load the forecast again to replace them.";
    }
    return true;
  }

  async loadForecast() {
    const fields = this.#elements;
    const button = fields.live;
    if (button.disabled) return;
    const revision = this.#editRevision;
    button.disabled = true;
    button.textContent = "Loading…";
    button.setAttribute("aria-busy", "true");
    fields.error.textContent = "";
    this.#clearForecast();
    try {
      const { weather, air } = await this.#forecasts.load();
      if (this.#isStale(revision)) return;
      const now = this.#now();
      const rows = forecastInputs(weather, air, fields.smoke.checked, now);
      const currentEpoch = Math.floor(now / MILLISECONDS_PER_HOUR) * SECONDS_PER_HOUR;
      const current = rows.find((input) => input.epoch === currentEpoch);
      if (!current) throw new Error("Current-hour weather or AQI is missing.");
      const best = this.#model.selectBestHour(rows);
      for (const id of [...CONDITION_IDS, "hour"]) fields[id].value = current[id];
      fields.weekend.checked = current.weekend;
      this.render();
      const forecastTime = best && FORECAST_TIME.format(new Date(best.epoch * 1000));
      fields.besthour.textContent = best
        ? `Lowest estimated chance of low demand in the next 24 hours, within study limits: ${forecastTime} (${percent(best.probability)}, AQI ${Math.round(best.aqi)}).`
        : "No complete daytime forecast hour meets the study's heat and AQI limits.";
      fields.forecastnote.textContent = "Open-Meteo weather and CAMS AQI forecasts. Hours from 06:00 to 21:00 DC time are compared. Smoke/haze uses your checkbox setting. Hours with missing data are excluded. Visibility is capped at 10 mi to match the training data.";
      button.textContent = "DC forecast loaded";
    } catch (error) {
      if (this.#isStale(revision)) return;
      this.#clearForecast();
      fields.error.textContent = `Forecast unavailable: ${error.message} Enter conditions manually.`;
    } finally {
      if (button.textContent === "Loading…") button.textContent = "Load DC forecast";
      button.disabled = false;
      button.removeAttribute("aria-busy");
    }
  }
}
