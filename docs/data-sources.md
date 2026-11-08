# Official dataset inventory

Checked 8 November 2026. These are legitimate provider access routes, **not confirmation that every proposed station-variable-year combination is available**. No observations have been downloaded.

## 1. HSY measurements through FMI — primary outcome source

- Provider: Helsinki Region Environmental Services (HSY); dissemination through the Finnish Meteorological Institute (FMI).
- [HSY explains its open data and FMI dissemination](https://www.hsy.fi/en/air-quality-and-climate/air-quality-in-the-helsinki-metropolitan-area/where-can-you-find-air-quality-information/).
- [FMI Download observations](https://en.ilmatieteenlaitos.fi/download-observations): free browser-based access to historical air-quality and weather observations.
- [Download help](https://en.ilmatieteenlaitos.fi/download-observations-questions).
- [FMI open-data API manual](https://en.ilmatieteenlaitos.fi/open-data-manual) for later scripted acquisition. Discover current stored queries and parameter names from the service rather than guessing identifiers.

Later access steps: select air-quality observations, locate Helsinki/Espoo/Vantaa stations, select NO₂ or PM2.5 and the historical interval, inspect temporal resolution and export options, and save the exact selections. Preserve hourly measurements if available so daily quality rules remain reproducible. If the browser does not expose the required series, check the API metadata or EEA verified archive.

Before selection, verify station ID, pollutant, units, flags, validation status, operation dates, relocations, instrument changes, and missingness. The HSY page currently reports disruption to some real-time updates; that does not establish whether historical exports are affected.

## 2. FMI weather — covariates

Use the same [download service](https://en.ilmatieteenlaitos.fi/download-observations) and [manual](https://en.ilmatieteenlaitos.fi/open-data-manual). Candidate measurements are wind speed, air temperature, and precipitation. Pick an appropriate nearby weather station only after checking historical availability and location; do not assume every air-quality site has colocated weather.

Check observation versus aggregated values, precipitation accumulation windows, time zones, and units. Record distances and explain the limitations of assigning one weather station to several air-quality stations.

## 3. EEA — verified historical fallback

[European Environment Agency Air Quality download service](https://www.eea.europa.eu/en/datahub/datahubitem-view/datahubitem-view/778ef9f5-6293-4846-badd-56a29c70880d) provides reported measurement series from 2013 onward, distributed as zipped Parquet files.

Select Finland, the required pollutant, stations, year, and **verified E1a** data where available; retain the selection metadata and quality flags. Up-to-date E2a is a different, unverified product. Confirm metadata and licence on the selected download. Do not append duplicate EEA and FMI observations of the same station-time.

## 4. HSY station context — metadata, not the outcome table

[HSY monitoring-site open-data description](https://www.hsy.fi/en/environmental-information/open-data/avoin-data---sivut/real-time-air-quality-at-hsy-monitoring-sites-/) describes station environments and permanent/mobile sites. Its real-time WMS/WFS layer contains an air-quality index; it is not a historical PM2.5/NO₂ concentration archive. Use site context to justify station selection and confirm historical station identity separately.

## Licensing and attribution

[FMI licence and terms](https://www.ilmatieteenlaitos.fi/avoin-data-lisenssi) specify CC BY 4.0 for covered open data, including air-quality data. Identify the original licensor/provider for each selected series, retain its metadata, credit both provider and access service as appropriate, and state any transformations. Check HSY and EEA product-specific terms separately; do not assume a licence applies to every map or third-party product.

Attribution template: “Source: [original provider], [dataset/series title], accessed through [service], retrieved [date], [licence URL]. Aggregation and other changes: [details].”

## Feasibility gate — group to complete before analysis

- [ ] One pollutant, exact year, stable station IDs, and at least three suitable sites confirmed.
- [ ] Measurement units, quality/validation flags, time basis, and nonpositive-value conventions understood.
- [ ] Weather coverage and matching rule confirmed.
- [ ] Missingness and station-days sufficient for the proposed comparison.
- [ ] Exact export/query, licence, attribution, and checksums recorded.
- [ ] Existing studies checked; explain how this question/workflow differs.
