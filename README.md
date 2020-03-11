# flower_app

### Prototype a Machine Learning model using Azle and Jupyter Kernel Gateway.

How to run in local:

1. Clone the repo

2. Launch jupyter kernel gateway within the project folder to expose GET endpoints on http://localhost:9090/

```
jupyter kernelgateway --KernelGatewayApp.api='kernel_gateway.notebook_http' --KernelGatewayApp.ip=0.0.0.0 --KernelGatewayApp.port=9090 --KernelGatewayApp.seed_uri=flower_app.ipynb --KernelGatewayApp.allow_origin='*'
```

3. Launch http server 

```
python -m http.server
```

4. Go to http://localhost:8000/


