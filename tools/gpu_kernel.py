"""Jupyter kernel launcher sharing the terminal runner's memory limits."""
from runtime import managed_entrypoint

managed_entrypoint()
from ipykernel.kernelapp import IPKernelApp

# JVM logs and HEARTBEAT go to the kernel's stderr (terminal log, or VS Code's Jupyter output channel),
# never into cell outputs of the submitted notebook.
IPKernelApp.launch_instance(capture_fd_output=False)
