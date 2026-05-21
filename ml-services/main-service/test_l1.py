import urllib.request, json, io

boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
body = io.BytesIO()

def add_field(name, value):
    body.write(b"--" + boundary.encode() + b"\r\n")
    body.write(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
    body.write(value.encode() + b"\r\n")

def add_file(name, filename):
    body.write(b"--" + boundary.encode() + b"\r\n")
    body.write(f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode())
    body.write(b"Content-Type: text/csv\r\n\r\n")
    with open(filename, "rb") as f:
        body.write(f.read())
    body.write(b"\r\n")

add_file("file", "iris_test.csv")
add_field("target_column", "species")
add_field("hyperparameters", '{"penalty":"l1","C":0.5,"max_iter":200}')
body.write(b"--" + boundary.encode() + b"--\r\n")

req = urllib.request.Request(
    "http://localhost:8000/models/logistic_regression/train",
    data=body.getvalue(),
    headers={"Content-Type": "multipart/form-data; boundary=" + boundary},
)
resp = urllib.request.urlopen(req)
result = json.loads(resp.read())
print("Accuracy:", result.get("metrics", {}).get("accuracy"))
print("Solver used:", result.get("model_params", {}).get("solver"))
print(json.dumps(result, indent=2)[:800])
