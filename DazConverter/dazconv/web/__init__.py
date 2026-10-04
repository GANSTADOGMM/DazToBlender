"""The local web application: a job queue and a dependency-free HTTP server."""

from .jobs import Job, JobQueue
from .server import serve

__all__ = ["Job", "JobQueue", "serve"]
