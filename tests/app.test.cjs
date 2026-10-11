const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { spawnSync } = require("node:child_process");
const test = require("node:test");
const vm = require("node:vm");

const root = path.resolve(__dirname, "..");
const appRoot = path.join(root, "app");
const fixedNow = Date.parse("2026-10-10T14:20:00Z");
const currentEpoch = Math.floor(fixedNow / 3600000) * 3600;

function app(fetcher = async () => { throw new Error("offline"); }) {
  const elements = {};
  const html = fs.readFileSync(path.join(root, "index.html"), "utf8");
  const definitions = [...html.matchAll(/<input\b([^>]+)>/g)];
  class Element {
    constructor(id, attrs = {}) {
      this.id = id;
      this.value = attrs.value || "";
      this.min = attrs.min || "";
      this.max = attrs.max || "";
      this.checked = false;
      this.disabled = false;
      this.hidden = true;
      this.textContent = "";
      this.attributes = {};
      this.listeners = {};
      this.children = [];
      this.className = "";
    }
    get valueAsNumber() { return String(this.value).trim() === "" ? NaN : Number(this.value); }
    checkValidity() {
      return Number.isFinite(this.valueAsNumber) &&
        (this.min === "" || this.valueAsNumber >= Number(this.min)) &&
        (this.max === "" || this.valueAsNumber <= Number(this.max));
    }
    setAttribute(name, value) { this.attributes[name] = value; }
    removeAttribute(name) { delete this.attributes[name]; }
    closest() { return { firstChild: { textContent: this.id } }; }
    addEventListener(name, handler) { this.listeners[name] = handler; }
    appendChild(child) { this.children.push(child); }
  }
  for (const [, attrs] of definitions) {
    const fields = Object.fromEntries([...attrs.matchAll(/(\w+)="([^"]*)"/g)].map((m) => [m[1], m[2]]));
    elements[fields.id] = new Element(fields.id, fields);
  }
  for (const id of ["form", "hour", "live", "error", "result", "risk", "pct", "band", "reco", "detail", "besthour", "forecastnote", "meta", "modelnote"]) {
    elements[id] = new Element(id);
  }
  const inputs = ["temp", "humidity", "aqi", "wind", "precip", "visibility", "hour", "weekend", "smoke"];
  class Clock extends Date { static now() { return fixedNow; } }
  const context = vm.createContext({
    window: {}, Date: Clock, Intl, AbortController, setTimeout, clearTimeout,
    fetch: fetcher,
    document: {
      getElementById: (id) => elements[id],
      querySelectorAll: () => inputs.map((id) => elements[id]),
      createElement: () => new Element("option"),
    },
  });
  vm.runInContext(fs.readFileSync(path.join(appRoot, "model.js"), "utf8"), context);
  vm.runInContext(fs.readFileSync(path.join(appRoot, "app.js"), "utf8"), context);
  return { context, elements, run: (code) => vm.runInContext(code, context) };
}

function forecast(overrides = {}) {
  const times = [currentEpoch, currentEpoch + 3600, currentEpoch + 7200];
  const weather = {
    hourly_units: { time: "unixtime", temperature_2m: "°F", relative_humidity_2m: "%", wind_speed_10m: "mp/h", precipitation: "inch", visibility: "ft" },
    hourly: { time: times, temperature_2m: [75, 78, 80], relative_humidity_2m: [60, 60, 60], wind_speed_10m: [0, 8, 9], precipitation: [0.24, 0.1, 0], visibility: [15840, 52800, 79200] },
  };
  const air = { hourly_units: { time: "unixtime", us_aqi: "USAQI" }, hourly: { time: times, us_aqi: [42, 70, 160] } };
  Object.assign(weather.hourly, overrides);
  return { weather, air };
}

function installForecast(context, data) {
  context.weather = data.weather;
  context.air = data.air;
  context.now = fixedNow;
}

function fetchForecast(data) {
  return async (url) => ({ ok: true, json: async () => url.includes("air-quality-api") ? data.air : data.weather });
}

test("published app and results pages reference existing local assets", () => {
  for (const entry of ["index.html", "results/index.html"]) {
    const htmlPath = path.join(root, entry);
    const html = fs.readFileSync(htmlPath, "utf8");
    const references = [...html.matchAll(/<(?:script|link|img)\b[^>]*(?:src|href)="([^"]+)"/g)]
      .map((match) => match[1])
      .filter((reference) => !/^(?:[a-z][a-z0-9+.-]*:|\/\/|#)/i.test(reference));
    assert.ok(references.length > 0, `${entry} must load local assets`);
    for (const reference of references) {
      const asset = path.resolve(path.dirname(htmlPath), reference.split(/[?#]/, 1)[0]);
      assert.ok(fs.statSync(asset).isFile(), `missing asset in ${entry}: ${reference}`);
    }
  }
});

test("shipped JavaScript features and scores match Python across heat regimes", () => {
  const cases = [];
  for (const [temp, humidity] of [[-40, 0], [8, 0], [37, 100], [30, 60], [75, 30], [79, 90], [80, 86], [86, 90], [100, 39], [100, 10], [112, 5], [95, 70]]) {
    for (const smoke of [false, true]) cases.push({ temp, humidity, aqi: 76, wind: 7.5, precip: 0.23, visibility: 4.2, hour: 19, weekend: true, smoke });
  }
  const python = spawnSync(process.env.PULSESHIFT_PYTHON || "python3", ["-c", `
import json, math, pathlib, sys
import numpy as np
from pulseshift.features import heat_index_f
model = json.loads(pathlib.Path('app/model.js').read_text().split('=', 1)[1].strip().removesuffix(';'))
out = []
for x in json.load(sys.stdin):
    hi = float(heat_index_f(x['temp'], x['humidity']))
    f = dict(heat_index_f=hi, cold_stress=max(0, model['stress']['cold_base_f']-x['temp']), heat_stress=max(0, hi-model['stress']['heat_base_f']), aqi=x['aqi'], humidity=x['humidity'], wind_mph=x['wind'], precip_in=x['precip'], visibility_mi=x['visibility'], smoke_haze=int(x['smoke']), hour_sin=math.sin(2*math.pi*x['hour']/24), hour_cos=math.cos(2*math.pi*x['hour']/24), is_weekend=int(x['weekend']))
    z = model['intercept'] + sum(c*(f[k]-m)/s for k,c,m,s in zip(model['features'], model['coef'], model['mean'], model['scale']))
    out.append([hi, 1/(1+np.exp(-z))])
print(json.dumps(out))
`], { cwd: root, env: { ...process.env, PYTHONPATH: path.join(root, "research") }, input: JSON.stringify(cases), encoding: "utf8" });
  assert.equal(python.status, 0, python.stderr);
  const expected = JSON.parse(python.stdout);
  const instance = app();
  cases.forEach((input, i) => {
    instance.context.input = input;
    assert.equal(instance.run("heatIndex(input.temp, input.humidity)"), expected[i][0]);
    assert.ok(Math.abs(instance.run("predictSuppression(input)") - expected[i][1]) < 1e-12);
  });
  assert.equal(instance.run("heatIndex(100, 39)"), 108.5);
});

test("blank and out-of-range inputs stop scoring without hidden substitution", () => {
  const { elements: e, run } = app();
  e.temp.value = "";
  run("update()");
  assert.equal(e.result.hidden, true);
  assert.match(e.error.textContent, /temp/);
  assert.equal(e.temp.attributes["aria-invalid"], "true");
  e.temp.value = "200";
  run("update()");
  assert.equal(e.result.hidden, true);
  assert.equal(e.temp.value, "200");
  e.temp.value = "-5";
  run("update()");
  assert.equal(e.result.hidden, false);
  assert.equal(e.error.textContent, "");
});

test("smoke and measured visibility are independent model inputs", () => {
  const { run, elements: e } = app();
  e.smoke.checked = true;
  e.visibility.value = "8.5";
  assert.equal(run("features(readInputs()).visibility_mi"), 8.5);
  assert.equal(run("features(readInputs()).smoke_haze"), 1);
});

test("HTTP error responses are rejected before parsing", async () => {
  const { run } = app(async () => ({ ok: false, status: 503, json: () => { throw new Error("must not parse"); } }));
  await assert.rejects(run("fetchJson('https://api.open-meteo.com/')"), /HTTP 503/);
});

test("forecast units convert feet and metres and retain real hourly rain", () => {
  const { context, run } = app();
  const data = forecast();
  installForecast(context, data);
  assert.equal(run("forecastInputs(weather, air, false, now)[0].visibility"), 3);
  assert.equal(run("forecastInputs(weather, air, false, now)[0].precip"), 0.24);
  assert.equal(run("forecastInputs(weather, air, false, now)[2].visibility"), 10);
  data.weather.hourly_units.visibility = "m";
  data.weather.hourly.visibility[0] = 4828.032;
  assert.equal(run("forecastInputs(weather, air, false, now)[0].visibility"), 3);
  data.weather.hourly_units.temperature_2m = "°C";
  assert.throws(() => run("forecastInputs(weather, air, false, now)"), /units/);
});

test("missing weather or AQI excludes a forecast hour instead of supplying defaults", () => {
  const { context, run } = app();
  const data = forecast({ relative_humidity_2m: [null, 60, 60] });
  data.air.hourly.us_aqi[1] = null;
  installForecast(context, data);
  assert.equal(run("forecastInputs(weather, air, false, now).length"), 1);
  assert.equal(run("forecastInputs(weather, air, false, now)[0].epoch"), currentEpoch + 7200);
});

test("timestamp joins distinguish repeated DC daylight-saving hours", () => {
  const { context, run } = app();
  const first = Date.parse("2026-11-01T05:00:00Z") / 1000;
  const second = first + 3600;
  assert.equal(run(`dcHourWeekend(${first}).hour`), 1);
  assert.equal(run(`dcHourWeekend(${second}).hour`), 1);
  assert.equal(run(`dcHourWeekend(${first}).weekend`), true);
  const data = forecast();
  data.weather.hourly.time = [first, second, second + 3600];
  data.air.hourly.time = [first, second, second + 3600];
  data.air.hourly.us_aqi = [12, 120, 42];
  installForecast(context, data);
  context.now = first * 1000;
  assert.equal(run("forecastInputs(weather, air, false, now)[0].aqi"), 12);
  assert.equal(run("forecastInputs(weather, air, false, now)[1].aqi"), 120);
});

test("forecast ranking excludes heat, AQI and non-daytime threshold failures", () => {
  const { context, run } = app();
  installForecast(context, forecast());
  context.rows = run("forecastInputs(weather, air, false, now)");
  context.rows[0].temp = 100;
  context.rows[0].humidity = 39;
  assert.equal(run("bestForecastHour(rows).epoch"), currentEpoch + 3600);
  context.rows[1].hour = 3;
  assert.equal(run("bestForecastHour(rows)"), null);
});

test("loading DC forecast updates every measured input and keeps explicit smoke setting", async () => {
  const instance = app(fetchForecast(forecast()));
  const e = instance.elements;
  e.precip.value = "3";
  e.hour.value = "1";
  e.smoke.checked = true;
  await instance.run("liveWeather()");
  assert.equal(Number(e.precip.value), 0.24);
  assert.equal(Number(e.hour.value), 10);
  assert.equal(Number(e.wind.value), 0);
  assert.equal(Number(e.visibility.value), 3);
  assert.equal(e.weekend.checked, true);
  assert.equal(e.smoke.checked, true);
  assert.match(e.besthour.textContent, /Oct 10/);
  assert.match(e.besthour.textContent, /EDT/);
  assert.match(e.forecastnote.textContent, /Smoke\/haze uses your checkbox/);
  assert.equal(e.live.disabled, false);
  assert.equal(e.live.attributes["aria-busy"], undefined);
});

test("unavailable current AQI preserves manual inputs and shows the missing-data error", async () => {
  const data = forecast();
  data.air.hourly.us_aqi[0] = null;
  const { elements: e, run } = app(fetchForecast(data));
  await run("liveWeather()");
  assert.equal(e.temp.value, "85");
  assert.match(e.error.textContent, /Current-hour weather or AQI is missing/);
  assert.equal(e.besthour.textContent, "");
});

test("failed forecast requests do not replace manual inputs or claim they loaded", async () => {
  const { elements: e, run } = app();
  await run("liveWeather()");
  assert.equal(e.temp.value, "85");
  assert.equal(e.live.textContent, "Load DC forecast");
  assert.match(e.error.textContent, /Forecast unavailable/);
});

test("manual edits made during a request survive its completion", async () => {
  const data = forecast();
  const pending = [];
  const { elements: e, run } = app((url) => new Promise((resolve) => pending.push(() => resolve({ ok: true, json: async () => url.includes("air-quality-api") ? data.air : data.weather }))));
  const loading = run("liveWeather()");
  e.temp.value = "94";
  e.temp.listeners.input();
  pending.forEach((resolve) => resolve());
  await loading;
  assert.equal(e.temp.value, "94");
  assert.match(e.error.textContent, /Inputs changed during loading/);
  assert.equal(e.besthour.textContent, "");
});

test("a repeated load click cannot start overlapping forecast requests", async () => {
  const pending = [];
  const data = forecast();
  const { run } = app((url) => new Promise((resolve) => pending.push(() => resolve({ ok: true, json: async () => url.includes("air-quality-api") ? data.air : data.weather }))));
  const first = run("liveWeather()");
  await run("liveWeather()");
  assert.equal(pending.length, 2);
  pending.forEach((resolve) => resolve());
  await first;
});

test("Enter submits without navigation and manual edits clear forecast ranking", () => {
  const { elements: e } = app();
  let prevented = false;
  e.form.listeners.submit({ preventDefault() { prevented = true; } });
  assert.equal(prevented, true);
  e.besthour.textContent = "previous forecast";
  e.forecastnote.textContent = "previous source";
  e.humidity.listeners.input();
  assert.equal(e.besthour.textContent, "");
  assert.equal(e.forecastnote.textContent, "");
});

test("displayed probability describes aggregate demand and threshold exceedance", () => {
  const { elements: e, run } = app();
  assert.match(e.band.textContent, /low bike-share demand/);
  assert.match(e.reco.textContent, /network rides/);
  e.temp.value = "100";
  e.humidity.value = "39";
  run("update()");
  assert.equal(e.band.textContent, "Study threshold exceeded");
  assert.match(e.reco.textContent, /official advisories/);
  assert.match(e.modelnote.textContent, /not personal exercise risk/);
  assert.equal(run("percent(0)"), "<1%");
  assert.equal(run("percent(0.009)"), "<1%");
  assert.equal(run("percent(0.01)"), "1%");
  assert.equal(run("percent(0.5)"), "50%");
  assert.equal(run("percent(0.99)"), "99%");
  assert.equal(run("percent(0.991)"), ">99%");
  assert.equal(run("percent(1)"), ">99%");
});

test("conflicting AQI timestamps are rejected regardless of record order", () => {
  const { context, run } = app();
  for (const values of [[200, 42], [42, 200]]) {
    const data = forecast();
    data.air.hourly.time = [currentEpoch, currentEpoch, currentEpoch + 3600, currentEpoch + 7200];
    data.air.hourly.us_aqi = [...values, 70, 160];
    installForecast(context, data);
    assert.throws(() => run("forecastInputs(weather, air, false, now)"), /duplicate timestamps/);
  }
});

test("conflicting weather timestamps are rejected before ranking", () => {
  const { context, run } = app();
  const data = forecast();
  data.weather.hourly.time = [...data.weather.hourly.time];
  for (const field of Object.keys(data.weather.hourly)) {
    data.weather.hourly[field].splice(1, 0, data.weather.hourly[field][0]);
  }
  data.weather.hourly.temperature_2m[1] = 100;
  data.weather.hourly.relative_humidity_2m[1] = 39;
  installForecast(context, data);
  assert.throws(() => run("forecastInputs(weather, air, false, now)"), /duplicate timestamps/);
  for (const field of Object.keys(data.weather.hourly)) data.weather.hourly[field].reverse();
  assert.throws(() => run("forecastInputs(weather, air, false, now)"), /duplicate timestamps/);
});

test("ambiguous forecasts preserve manual inputs and do not publish a recommendation", async () => {
  const data = forecast();
  data.air.hourly.time = [...data.air.hourly.time, currentEpoch];
  data.air.hourly.us_aqi.push(200);
  const { elements: e, run } = app(fetchForecast(data));
  e.temp.value = "91";
  e.aqi.value = "55";
  await run("liveWeather()");
  assert.equal(e.temp.value, "91");
  assert.equal(e.aqi.value, "55");
  assert.match(e.error.textContent, /duplicate timestamps/);
  assert.equal(e.besthour.textContent, "");
  assert.equal(e.forecastnote.textContent, "");
  assert.equal(e.live.textContent, "Load DC forecast");
});
