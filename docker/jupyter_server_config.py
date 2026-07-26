# PyCharm Docker Compose starts jupyterlab without --ip and cannot scrape
# a redacted token from docker logs (token=...), so leave auth open for local IDE use.
# The compose CMD may still set --ServerApp.token for browser access on :8888.
c.ServerApp.ip = "0.0.0.0"
c.ServerApp.allow_remote_access = True
c.ServerApp.token = ""
c.ServerApp.password = ""
c.IdentityProvider.token = ""
