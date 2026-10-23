const appUrl = new URL("app/", window.location.href);
appUrl.search = window.location.search;
appUrl.hash = window.location.hash;
window.location.replace(appUrl);
