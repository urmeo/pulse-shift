try {
  const [{ PulseShiftApp }, { DemandModel }, { ForecastClient }] = await Promise.all([
    import("./application.mjs"),
    import("./demand-model.mjs"),
    import("./forecast-client.mjs"),
  ]);
  const app = new PulseShiftApp(document, {
    model: new DemandModel(window.PULSESHIFT_MODEL),
    forecastClient: new ForecastClient(),
  });
  app.start();
} catch {
  document.getElementById("error").textContent = "The demand model could not be loaded. Reload the page.";
  document.getElementById("live").disabled = true;
}
