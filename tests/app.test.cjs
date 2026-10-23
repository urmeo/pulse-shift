const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { spawnSync } = require("node:child_process");
const test = require("node:test");

const repoRoot = path.resolve(__dirname, "..");
const appDirectory = path.join(repoRoot, "app");
const modelArtifact = JSON.parse(fs.readFileSync(path.join(appDirectory, "model.js"), "utf8")
  .replace(/^window\.PULSESHIFT_MODEL\s*=\s*/, "").replace(/;\s*$/, ""));
const forecast = require("./fixtures/forecast.json");
const elementAttributes = [...fs.readFileSync(path.join(appDirectory, "index.html"), "utf8")
  .matchAll(/<[^>]+\bid="[^"]+"[^>]*>/g)].map(([tag]) => Object.fromEntries(
    [...tag.matchAll(/([\w-]+)="([^"]*)"/g)].map((match) => [match[1], match[2]])
  ));
const fixedNowMs = Date.parse("2026-10-10T14:20:00Z");
const currentHourEpoch = Math.floor(fixedNowMs / 3600000) * 3600;
const conditions = {
  temp: 85, humidity: 55, aqi: 60, wind: 6, precip: 0, visibility: 10,
  hour: 17, weekend: false, smoke: false,
};

function makeForecast() {
  return structuredClone(forecast);
}

function forecastResponse(data, url) {
  return { ok: true, json: async () => url.includes("air-quality-api") ? data.air : data.weather };
}

function deferredForecast(error) {
  const data = makeForecast();
  const pending = [];
  return {
    fetcher: (url) => new Promise((resolve, reject) => {
      pending.push(() => error ? reject(error) : resolve(forecastResponse(data, url)));
    }),
    get requestCount() { return pending.length; },
    complete() { pending.forEach((complete) => complete()); },
  };
}

let DemandModel, heatIndex, ForecastClient, forecastInputs, dcHourWeekend, PulseShiftApp, percent, model;
test.before(async () => {
  ({ DemandModel, heatIndex } = await import("../app/demand-model.mjs"));
  ({ ForecastClient, forecastInputs, dcHourWeekend } = await import("../app/forecast-client.mjs"));
  ({ PulseShiftApp, percent } = await import("../app/application.mjs"));
  model = new DemandModel(modelArtifact);
});

function waitForAbort(signal, onAbort) {
  return new Promise((_, reject) => {
    signal.addEventListener("abort", () => {
      onAbort();
      reject(signal.reason);
    }, { once: true });
  });
}

class Element {
  constructor(attributes) {
    Object.assign(this, { value: "", min: "", max: "", checked: false, disabled: false, textContent: "" }, attributes);
    this.attributes = {};
    this.listeners = {};
  }
  get valueAsNumber() {
    return String(this.value).trim() === "" ? NaN : Number(this.value);
  }
  checkValidity() {
    return Number.isFinite(this.valueAsNumber) &&
      (this.min === "" || this.valueAsNumber >= Number(this.min)) &&
      (this.max === "" || this.valueAsNumber <= Number(this.max));
  }
  setAttribute(name, value) { this.attributes[name] = value; }
  removeAttribute(name) { delete this.attributes[name]; }
  addEventListener(name, handler) { this.listeners[name] = handler; }
  appendChild() {}
}

function makeApp(fetcher = async () => { throw new Error("offline"); }) {
  const elements = Object.fromEntries(elementAttributes.map((attributes) => [
    attributes.id, new Element(attributes),
  ]));
  let now = fixedNowMs;
  const app = new PulseShiftApp({
    getElementById: (id) => elements[id],
    createElement: () => ({}),
  }, {
    model,
    forecastClient: new ForecastClient({ fetcher }),
    now: () => now,
  });
  app.start();
  return { app, elements, setNow(value) { now = value; } };
}

test("shipped JavaScript features and scores match Python across heat regimes", () => {
  const cases = [
    [-40, 0], [8, 0], [37, 100], [30, 60], [75, 30], [79, 90],
    [80, 86], [86, 90], [100, 39], [100, 10], [112, 5], [95, 70],
  ].flatMap(([temp, humidity]) => [false, true].map((smoke) => ({
    temp, humidity, aqi: 76, wind: 7.5, precip: 0.23, visibility: 4.2, hour: 19, weekend: true, smoke,
  })));
  const python = spawnSync(process.env.PULSESHIFT_PYTHON || "python3", ["-c", `
import json
import math
from pathlib import Path
import sys

import numpy as np

from pulseshift.features import heat_index_f

model_text = Path("app/model.js").read_text().split("=", 1)[1]
model = json.loads(model_text.strip().removesuffix(";"))
predictions = []
for inputs in json.load(sys.stdin):
    heat_index = float(heat_index_f(inputs["temp"], inputs["humidity"]))
    features = {
        "heat_index_f": heat_index,
        "cold_stress": max(0, model["stress"]["cold_base_f"] - inputs["temp"]),
        "heat_stress": max(0, heat_index - model["stress"]["heat_base_f"]),
        "aqi": inputs["aqi"],
        "humidity": inputs["humidity"],
        "wind_mph": inputs["wind"],
        "precip_in": inputs["precip"],
        "visibility_mi": inputs["visibility"],
        "smoke_haze": int(inputs["smoke"]),
        "hour_sin": math.sin(2 * math.pi * inputs["hour"] / 24),
        "hour_cos": math.cos(2 * math.pi * inputs["hour"] / 24),
        "is_weekend": int(inputs["weekend"]),
    }
    score = model["intercept"] + sum(
        coefficient * (features[name] - mean) / scale
        for name, coefficient, mean, scale in zip(
            model["features"], model["coef"], model["mean"], model["scale"]
        )
    )
    predictions.append([heat_index, 1 / (1 + np.exp(-score))])
print(json.dumps(predictions))
`], {
    cwd: repoRoot,
    env: { ...process.env, PYTHONPATH: path.join(repoRoot, "research") },
    input: JSON.stringify(cases),
    encoding: "utf8",
  });
  assert.equal(python.status, 0, python.stderr);
  const expected = JSON.parse(python.stdout);
  cases.forEach((input, index) => {
    assert.equal(heatIndex(input.temp, input.humidity), expected[index][0]);
    assert.ok(Math.abs(model.evaluate(input).probability - expected[index][1]) < 1e-12);
  });
  for (const [temp, humidity, expected] of [[100, 39, 108.5], [8, 0, 3.2], [-40, 0, -47.2]]) {
    assert.equal(heatIndex(temp, humidity), expected);
  }
});

for (const [field, invalidValues, correction, message] of [
  ["temp", ["", "200"], "-5", /Temperature/],
  ["hour", ["", "24", "12.5"], "0", /Select a start hour/],
]) {
  test(`${field} rejects invalid values and recovers without substitution`, () => {
    const { elements, app } = makeApp();
    for (const value of invalidValues) {
      elements[field].value = value;
      elements[field].listeners.input();
      assert.equal(elements.result.hidden, true);
      assert.equal(elements[field].value, value);
      assert.equal(elements[field].attributes["aria-invalid"], "true");
      assert.match(elements.error.textContent, message);
    }
    elements[field].value = correction;
    elements[field].listeners.input();
    assert.equal(elements[field].attributes["aria-invalid"], "false");
    assert.equal(app.readInputs()[field], Number(correction));
    assert.equal(elements.result.hidden, false);
    assert.equal(elements.error.textContent, "");
  });
}

test("smoke and measured visibility are independent model inputs", () => {
  const { app, elements } = makeApp();
  elements.smoke.checked = true;
  elements.visibility.value = "8.5";
  assert.equal(model.features(app.readInputs()).visibility_mi, 8.5);
  assert.equal(model.features(app.readInputs()).smoke_haze, 1);
});

test("corrected fields lose stale invalid markers while another field is invalid", () => {
  const { elements, app } = makeApp();
  elements.aqi.value = "";
  app.render();
  assert.equal(elements.aqi.attributes["aria-invalid"], "true");
  elements.temp.value = "";
  elements.aqi.value = "60";
  elements.aqi.listeners.input();
  assert.equal(elements.temp.attributes["aria-invalid"], "true");
  assert.equal(elements.aqi.attributes["aria-invalid"], "false");
  assert.equal(elements.result.hidden, true);
  assert.match(elements.error.textContent, /Temperature/);
  elements.temp.value = "85";
  elements.temp.listeners.input();
  assert.equal(elements.result.hidden, false);
  assert.equal(elements.error.textContent, "");
});

for (const [name, alter, error] of [
  ["missing weather body", (data) => { data.weather = null; }, /Unexpected forecast units or fields/],
  ["missing AQI body", (data) => { data.air = null; }, /Unexpected forecast units or fields/],
  ["missing current AQI", (data) => { data.air.hourly.us_aqi[0] = null; }, /Current-hour weather or AQI is missing/],
  ["failed request", null, /Forecast unavailable/],
  ["ambiguous forecast", (data) => {
    data.air.hourly.time.push(currentHourEpoch);
    data.air.hourly.us_aqi.push(200);
  }, /duplicate timestamps/],
]) {
  test(`${name} preserves manual inputs and clears forecast output`, async () => {
    const data = makeForecast();
    alter?.(data);
    const { elements, app } = makeApp(alter ? (url) => forecastResponse(data, url) : undefined);
    elements.temp.value = "91";
    elements.aqi.value = "55";
    await app.loadForecast();
    assert.equal(elements.temp.value, "91");
    assert.equal(elements.aqi.value, "55");
    assert.match(elements.error.textContent, error);
    assert.equal(elements.besthour.textContent, "");
    assert.equal(elements.forecastnote.textContent, "");
    assert.equal(elements.live.disabled, false);
    assert.equal(elements.live.textContent, "Load DC forecast");
  });
}

test("forecast selection uses the current hour when a request crosses an hour boundary", async () => {
  const request = deferredForecast();
  const { elements, app, setNow } = makeApp(request.fetcher);
  let now = (currentHourEpoch + 3600) * 1000 - 2000;
  setNow(now);
  const loading = app.loadForecast();
  now += 4000;
  setNow(now);
  request.complete();
  await loading;
  assert.equal(Number(elements.hour.value), 11);
  assert.equal(Number(elements.temp.value), 78);
  assert.equal(Number(elements.aqi.value), 70);
  assert.equal(elements.live.textContent, "DC forecast loaded");
});

test("manual edits made during a request survive its completion", async () => {
  const request = deferredForecast();
  const { elements, app } = makeApp(request.fetcher);
  const loading = app.loadForecast();
  elements.temp.value = "94";
  elements.temp.listeners.input();
  request.complete();
  await loading;
  assert.equal(elements.temp.value, "94");
  assert.match(elements.error.textContent, /Inputs changed during loading/);
  assert.equal(elements.besthour.textContent, "");
});

test("stale forecast success and failure preserve current validation errors", async () => {
  for (const failed of [false, true]) {
    const request = deferredForecast(failed ? new Error("offline") : undefined);
    const { elements, app } = makeApp(request.fetcher);
    const loading = app.loadForecast();
    elements.temp.value = "";
    elements.temp.listeners.input();
    const validationError = elements.error.textContent;
    request.complete();
    await loading;
    assert.equal(elements.error.textContent, validationError);
    assert.equal(elements.result.hidden, true);
    assert.equal(elements.temp.attributes["aria-invalid"], "true");
    assert.equal(elements.live.disabled, false);
    assert.equal(elements.live.attributes["aria-busy"], undefined);
  }
});

test("a repeated load click cannot start overlapping forecast requests", async () => {
  const request = deferredForecast();
  const { app } = makeApp(request.fetcher);
  const first = app.loadForecast();
  await app.loadForecast();
  assert.equal(request.requestCount, 2);
  request.complete();
  await first;
});

test("displayed probability describes aggregate demand and threshold exceedance", () => {
  const { elements, app } = makeApp();
  assert.match(elements.band.textContent, /low bike-share demand/);
  assert.match(elements.reco.textContent, /network rides/);
  elements.temp.value = "100";
  elements.humidity.value = "39";
  app.render();
  assert.equal(elements.band.textContent, "Study threshold exceeded");
  assert.match(elements.reco.textContent, /official advisories/);
  assert.match(elements.modelnote.textContent, /not personal exercise risk/);
  for (const [probability, label] of [
    [0, "<1%"], [0.009, "<1%"], [0.01, "1%"], [0.5, "50%"],
    [0.99, "99%"], [0.991, ">99%"], [1, ">99%"],
  ]) assert.equal(percent(probability), label);
});

test("model construction rejects malformed coefficients and feature definitions", () => {
  for (const alter of [
    (artifact) => { artifact.features[0] = "unknown"; },
    (artifact) => { artifact.features[0] = artifact.features[1]; },
    (artifact) => { artifact.features.pop(); },
    (artifact) => { artifact.mean.pop(); },
    (artifact) => { delete artifact.coef[0]; },
    (artifact) => { artifact.coef[0] = Infinity; },
    (artifact) => { artifact.mean[0] = NaN; },
    (artifact) => { artifact.scale[0] = 0; },
    (artifact) => { artifact.scale[0] = -1; },
    (artifact) => { artifact.scale[0] = Infinity; },
    (artifact) => { artifact.intercept = "0"; },
    (artifact) => { delete artifact.stress; },
    (artifact) => { artifact.stress.heat_base_f = NaN; },
    (artifact) => { delete artifact.safety; },
    (artifact) => { artifact.safety.aqi_unsafe = Infinity; },
  ]) {
    const artifact = structuredClone(modelArtifact);
    alter(artifact);
    assert.throws(() => new DemandModel(artifact), /Model artifact is invalid/);
  }
  for (const artifact of [null, undefined, {}, []]) {
    assert.throws(() => new DemandModel(artifact), /Model artifact is invalid/);
  }
});

test("model parameters are snapshotted and preserve feature ordering", () => {
  const artifact = structuredClone(modelArtifact);
  const model = new DemandModel(artifact);
  const original = model.evaluate(conditions);
  for (const field of ["features", "mean", "scale", "coef"]) artifact[field].reverse();
  const reordered = new DemandModel(artifact);
  assert.ok(Math.abs(reordered.evaluate(conditions).probability - original.probability) < 1e-15);
  artifact.coef.fill(0);
  artifact.intercept = 100;
  artifact.stress.heat_base_f = -100;
  artifact.safety.aqi_unsafe = 0;
  artifact.meta.auroc_2024 = 0;
  assert.deepEqual(model.evaluate(conditions), original);
  assert.equal(model.safety.aqi_unsafe, modelArtifact.safety.aqi_unsafe);
  assert.equal(model.meta.auroc_2024, modelArtifact.meta.auroc_2024);
  assert.ok(Object.isFrozen(model));
  assert.ok(Object.isFrozen(model.safety));
  assert.ok(Object.isFrozen(model.meta));
});

test("scoring rejects missing, nonnumeric and nonfinite inputs", () => {
  for (const [fields, invalidValues] of [
    [["temp", "humidity", "aqi", "wind", "precip", "visibility"], [undefined, null, "1", NaN, Infinity, -Infinity]],
    [["hour"], [-1, 24, 12.5, "17", NaN]],
    [["smoke", "weekend"], [undefined, null, 0, 1, "false"]],
  ]) {
    for (const field of fields) {
      for (const value of invalidValues) {
        assert.throws(() => model.evaluate({ ...conditions, [field]: value }), /Model inputs are invalid/);
      }
    }
  }
  assert.throws(() => model.evaluate(null), /Model inputs are invalid/);
});

test("feature and score overflow cannot produce a probability", () => {
  const artifact = structuredClone(modelArtifact);
  artifact.coef[0] = Number.MAX_VALUE;
  const model = new DemandModel(artifact);
  assert.throws(() => model.evaluate({ ...conditions, temp: 100, humidity: 90 }), /Model score is invalid/);
  assert.throws(() => model.evaluate({ ...conditions, temp: Number.MAX_VALUE }), /Model inputs are invalid/);
});

test("study thresholds exclude equality while allowing lower values", () => {
  const artifact = structuredClone(modelArtifact);
  artifact.safety.heat_unsafe_f = heatIndex(conditions.temp, conditions.humidity);
  const heatModel = new DemandModel(artifact);
  assert.equal(heatModel.evaluate(conditions).excluded, true);
  assert.equal(model.evaluate({ ...conditions, aqi: 150 }).excluded, true);
  assert.equal(model.evaluate({ ...conditions, aqi: 149.9 }).excluded, false);
});

test("hour selection chooses the lowest probability inside daytime study limits", () => {
  const row = (epoch, changes = {}) => ({ ...conditions, epoch, ...changes });
  const first = row(1000, { hour: 6, precip: 0.4 });
  const second = row(2000, { hour: 21 });
  const candidates = [
    row(3000, { hour: 5 }), row(4000, { hour: 22 }),
    row(5000, { aqi: 150 }), row(6000, { temp: 100, humidity: 39 }),
    first, second,
  ];
  const best = model.selectBestHour(candidates);
  assert.equal(best.epoch, second.epoch);
  assert.equal(best.probability, model.evaluate(second).probability);
  assert.equal(best.excluded, false);
  assert.equal(model.selectBestHour(candidates.slice(0, 4)), null);
  assert.equal(model.selectBestHour([]), null);
});

test("tied forecast probabilities select the earliest timestamp regardless of input order", () => {
  const earlier = { ...conditions, epoch: 1000 };
  const later = { ...conditions, epoch: 2000 };
  assert.equal(model.selectBestHour([later, earlier]).epoch, earlier.epoch);
  assert.equal(model.selectBestHour([earlier, later]).epoch, earlier.epoch);
  assert.throws(() => model.selectBestHour(null), /Forecast hours are invalid/);
  assert.throws(() => model.selectBestHour([{ ...conditions, epoch: NaN }]), /Forecast timestamp is invalid/);
});

test("forecast client loads both DC sources with matching hourly units", async () => {
  const data = makeForecast();
  const requests = [];
  const client = new ForecastClient({ fetcher: async (url, { signal }) => {
    requests.push({ url: new URL(url), signal });
    return forecastResponse(data, url);
  } });
  assert.deepEqual(await client.load(), data);
  assert.equal(requests.length, 2);
  assert.equal(requests[0].signal, requests[1].signal);
  assert.deepEqual(requests.map(({ url }) => url.hostname), [
    "api.open-meteo.com", "air-quality-api.open-meteo.com",
  ]);
  for (const { url } of requests) {
    assert.equal(url.searchParams.get("latitude"), "38.8951");
    assert.equal(url.searchParams.get("longitude"), "-77.0364");
    assert.equal(url.searchParams.get("timeformat"), "unixtime");
    assert.equal(url.searchParams.get("timezone"), "America/New_York");
    assert.equal(url.searchParams.get("forecast_hours"), "25");
  }
  assert.equal(requests[0].url.searchParams.get("temperature_unit"), "fahrenheit");
  assert.equal(requests[0].url.searchParams.get("wind_speed_unit"), "mph");
  assert.equal(requests[0].url.searchParams.get("precipitation_unit"), "inch");
  assert.equal(requests[1].url.searchParams.get("hourly"), "us_aqi");
});

for (const [name, response, message] of [
  ["HTTP error", { ok: false, status: 503 }, /HTTP 503/],
  ["invalid JSON", { ok: true }, /Invalid JSON/],
]) {
  test(`${name} cancels the pending sibling request`, async () => {
    let bodyParsed = false;
    let siblingAborted = false;
    const client = new ForecastClient({ fetcher: async (url, { signal }) => {
      if (url.includes("air-quality-api")) return waitForAbort(signal, () => { siblingAborted = true; });
      return { ...response, async json() { bodyParsed = true; throw new SyntaxError("Invalid JSON"); } };
    } });
    await assert.rejects(client.load(), message);
    assert.equal(bodyParsed, response.ok);
    assert.equal(siblingAborted, true);
  });
}

test("one timeout covers both requests and their response bodies", async () => {
  let bodiesAborted = 0;
  const client = new ForecastClient({
    timeoutMs: 10,
    fetcher: async (url, { signal }) => ({
      ok: true,
      json: () => waitForAbort(signal, () => { bodiesAborted++; }),
    }),
  });
  await assert.rejects(client.load(), /Forecast request timed out/);
  assert.equal(bodiesAborted, 2);
});

test("a failed load does not cancel the next attempt", async () => {
  const data = makeForecast();
  const signals = [];
  let failing = true;
  const client = new ForecastClient({ fetcher: async (url, { signal }) => {
    signals.push(signal);
    if (failing) throw new Error("offline");
    return forecastResponse(data, url);
  } });
  await assert.rejects(client.load(), /offline/);
  failing = false;
  assert.deepEqual(await client.load(), data);
  assert.notEqual(signals[0], signals[2]);
});

test("invalid client configuration fails before requesting data", () => {
  assert.throws(() => new ForecastClient({ fetcher: null }), /fetch function/);
  for (const timeoutMs of [0, -1, NaN, Infinity]) {
    assert.throws(() => new ForecastClient({ timeoutMs }), /timeout must be positive/);
  }
});

test("forecast parser joins UTC hours, converts visibility and retains measured rain", () => {
  const data = makeForecast();
  const rows = forecastInputs(data.weather, data.air, true, fixedNowMs);
  assert.equal(rows.length, 3);
  assert.deepEqual(rows.map((row) => row.visibility), [3, 10, 10]);
  assert.equal(rows[0].precip, 0.24);
  assert.equal(rows[0].wind, 0);
  assert.equal(rows[0].smoke, true);
  assert.equal(rows[0].hour, 10);
  assert.equal(rows[0].weekend, true);
  data.weather.hourly_units.visibility = "m";
  data.weather.hourly.visibility[0] = 4828.032;
  assert.equal(forecastInputs(data.weather, data.air, false, fixedNowMs)[0].visibility, 3);
});

test("forecast parser rejects incompatible units and incomplete arrays", () => {
  const wrongUnits = makeForecast();
  wrongUnits.weather.hourly_units.temperature_2m = "°C";
  assert.throws(() => forecastInputs(wrongUnits.weather, wrongUnits.air, false, fixedNowMs), /units or fields/);
  for (const field of Object.keys(makeForecast().weather.hourly)) {
    const data = makeForecast();
    data.weather.hourly[field] = [];
    assert.throws(() => forecastInputs(data.weather, data.air, false, fixedNowMs), /Incomplete forecast/);
  }
  const shortAqi = makeForecast();
  shortAqi.air.hourly.us_aqi.pop();
  assert.throws(() => forecastInputs(shortAqi.weather, shortAqi.air, false, fixedNowMs), /Incomplete forecast/);
});

test("forecast parser excludes missing and invalid measurements", () => {
  const data = makeForecast();
  data.weather.hourly.relative_humidity_2m[0] = null;
  data.air.hourly.us_aqi[1] = -1;
  const rows = forecastInputs(data.weather, data.air, false, fixedNowMs);
  assert.deepEqual(rows.map((row) => row.epoch), [currentHourEpoch + 7200]);
});

test("forecast parser excludes fractional timestamps before ranking valid hours", () => {
  const data = makeForecast();
  data.weather.hourly.time[2] += 0.5;
  data.air.hourly.time[2] += 0.5;
  const rows = forecastInputs(data.weather, data.air, false, fixedNowMs);
  assert.deepEqual(rows.map((row) => row.epoch), [currentHourEpoch, currentHourEpoch + 3600]);
  assert.equal(model.selectBestHour(rows).epoch, currentHourEpoch + 3600);
});

test("forecast parser rejects conflicting duplicate timestamps regardless of source or record order", () => {
  for (const source of ["weather", "air"]) {
    const data = makeForecast();
    for (const values of Object.values(data[source].hourly)) values.splice(1, 0, values[0]);
    if (source === "air") data.air.hourly.us_aqi[1] = 200;
    else data.weather.hourly.temperature_2m[1] = 100;
    assert.throws(() => forecastInputs(data.weather, data.air, false, fixedNowMs), /duplicate timestamps/);
    for (const values of Object.values(data[source].hourly)) values.reverse();
    assert.throws(() => forecastInputs(data.weather, data.air, false, fixedNowMs), /duplicate timestamps/);
  }
});

test("forecast parser preserves both DC hours in the daylight-saving fold", () => {
  const first = Date.parse("2026-11-01T05:00:00Z") / 1000;
  const data = makeForecast();
  data.weather.hourly.time = [first, first + 3600, first + 7200];
  data.air.hourly.time = [first, first + 3600, first + 7200];
  const rows = forecastInputs(data.weather, data.air, false, first * 1000);
  assert.deepEqual(dcHourWeekend(first), { hour: 1, weekend: true });
  assert.deepEqual(rows.map((row) => row.hour), [1, 1, 2]);
  assert.deepEqual(rows.map((row) => row.aqi), [42, 70, 160]);
});
