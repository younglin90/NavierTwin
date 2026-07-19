naviertwin.api
============================================================

.. automodule:: naviertwin.api
   :members:
   :undoc-members:
   :show-inheritance:

Server
------

.. automodule:: naviertwin.api.server
   :members:
   :undoc-members:
   :show-inheritance:

Operations
----------

Authentication is opt-in for local compatibility. Set comma-separated API
keys before exposing the service beyond localhost::

   export NAVIERTWIN_API_KEY_HASHES="$(printf %s 'replace-with-a-secret' | sha256sum | cut -d' ' -f1)"
   export NAVIERTWIN_RATE_LIMIT_REQUESTS=120
   export NAVIERTWIN_RATE_LIMIT_WINDOW_SECONDS=60
   export NAVIERTWIN_RATE_LIMIT_STORE=/var/lib/naviertwin/rate-limits.sqlite3
   export NAVIERTWIN_MAX_REQUEST_BYTES=67108864

Clients send either ``X-API-Key`` or ``Authorization: Bearer <key>``. Health
and readiness checks remain public. Every business endpoint is available under
``/api/v1``; unversioned routes remain compatibility aliases and return a
successor ``Link`` header. Operational endpoints include ``/api/v1/health``,
``/api/v1/ready``, ``/api/v1/metrics``, and
``/api/v1/metrics/prometheus``. Every HTTP response includes
``X-Request-ID``, ``X-API-Version``, and ``Server-Timing`` headers.

The SQLite rate-limit store is optional but required when multiple Uvicorn
workers must share one limit. Without it, each worker uses an in-memory sliding
window. Plaintext ``NAVIERTWIN_API_KEYS`` remains supported for development;
hashed keys are preferred.

Terminate TLS at a trusted reverse proxy or start Uvicorn with certificate
files. Proxy headers are disabled unless explicitly enabled::

   naviertwin server --host 0.0.0.0 --port 8443 \
     --workers 4 --ssl-certfile server.crt --ssl-keyfile server.key

.. automodule:: naviertwin.api.operations
   :members:
   :undoc-members:
   :show-inheritance:
