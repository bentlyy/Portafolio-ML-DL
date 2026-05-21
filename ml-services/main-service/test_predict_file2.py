import urllib.request, json, io

boundary = "----TestBoundary"
data = io.BytesIO()
# Add file
data.write(("--" + boundary + "\r\n").encode())
data.write('Content-Disposition: form-data; name="file"; filename="iris_test.csv"\r\n'.encode())
data.write("Content-Type: text/csv\r\n\r\n".encode())
with open("iris_test.csv", "rb") as f:
    data.write(f.read())
data.write(b"\r\n")
# Add target_column field
data.write(("--" + boundary + "\r\n").encode())
data.write('Content-Disposition: form-data; name="target_column"\r\n\r\n'.encode())
data.write(b"species\r\n")
data.write(("--" + boundary + "--\r\n").encode())

req = urllib.request.Request(
    "http://localhost:8000/models/random_forest/predict-from-file",
    data=data.getvalue(),
    headers={"Content-Type": "multipart/form-data; boundary=" + boundary},
)
try:
    resp = urllib.request.urlopen(req)
    result = json.loads(resp.read())
    print(json.dumps(result, indent=2)[:500])
except urllib.error.HTTPError as e:
    print("Error", e.code, ":", e.read().decode()[:300])
