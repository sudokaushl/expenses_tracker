ARG GRAFANA_IMAGE=grafana/grafana:11.6.1-ubuntu
FROM ${GRAFANA_IMAGE}

USER root
ARG DUCKDB_PLUGIN_VERSION=0.4.6
ARG DUCKDB_PLUGIN_SHA256=3464c8ee0a60eb49bdbd778caf103112dd710a234aa26e429faf813f449a8a57

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl unzip \
    && rm -rf /var/lib/apt/lists/* \
    && curl --fail --location --silent --show-error \
       "https://github.com/motherduckdb/grafana-duckdb-datasource/releases/download/v${DUCKDB_PLUGIN_VERSION}/motherduck-duckdb-datasource-${DUCKDB_PLUGIN_VERSION}.zip" \
       --output /tmp/duckdb-plugin.zip \
    && echo "${DUCKDB_PLUGIN_SHA256}  /tmp/duckdb-plugin.zip" | sha256sum --check \
    && mkdir -p /var/lib/grafana/plugins/motherduck-duckdb-datasource \
    && unzip -q /tmp/duckdb-plugin.zip -d /var/lib/grafana/plugins/motherduck-duckdb-datasource \
    && rm /tmp/duckdb-plugin.zip \
    && chown -R 472:0 /var/lib/grafana/plugins/motherduck-duckdb-datasource

USER 472
