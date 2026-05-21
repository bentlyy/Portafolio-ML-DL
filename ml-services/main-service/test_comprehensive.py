import urllib.request, json, io, sys

BASE = "http://localhost:8000"
passed = 0
failed = 0

def test(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  PASS: {name}")
        passed += 1
    else:
        print(f"  FAIL: {name} - {detail}")
        failed += 1

def req(method, path, body=None, files=None, form=None):
    if method == "GET":
        r = urllib.request.urlopen(f"{BASE}{path}")
        return json.loads(r.read())
    elif method == "POST":
        if files:
            boundary = "----TestBoundary"
            data = io.BytesIO()
            def add_field(name, value):
                data.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode())
            def add_file(name, filename):
                data.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; filename=\"{filename}\"\r\nContent-Type: text/csv\r\n\r\n".encode())
                with open(filename, "rb") as f:
                    data.write(f.read())
                data.write(b"\r\n")
            if form:
                for k, v in form.items():
                    add_field(k, v)
            for k, v in files.items():
                add_file(k, v)
            data.write(f"--{boundary}--\r\n".encode())
            req = urllib.request.Request(f"{BASE}{path}", data=data.getvalue(),
                headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
            r = urllib.request.urlopen(req)
            return json.loads(r.read())
        else:
            req = urllib.request.Request(f"{BASE}{path}",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"})
            r = urllib.request.urlopen(req)
            return json.loads(r.read())

print("=== 1. Health Check ===")
h = req("GET", "/health")
test("Status healthy", h["status"] == "healthy")
test("12 models available", h["models_available"] == 12)
test("4 categories", len(h["categories"]) == 4)

print("\n=== 2. Model Listing ===")
m = req("GET", "/models")
test("All 12 models returned", len(m["models"]) == 12)
test("Categories include classification", "classification" in m["categories"])
test("Categories include regression", "regression" in m["categories"])
test("Categories include clustering", "clustering" in m["categories"])
test("Categories include neural_network", "neural_network" in m["categories"])

print("\n=== 3. Model Info ===")
info = req("GET", "/models/random_forest")
test("Model ID matches", info["model_id"] == "random_forest")
test("Has hyperparameters", len(info["hyperparameters"]) > 0)

print("\n=== 4. Hyperparameters ===")
hp = req("GET", "/models/svm/hyperparameters")
test("Has model_id", hp["model_id"] == "svm")
test("Has hyperparameters dict", "hyperparameters" in hp)

print("\n=== 5. Upload Dataset ===")
# Test with iris
boundary = "----TestBoundary"
data = io.BytesIO()
def add_field(name, value):
    data.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode())
def add_file(name, filename):
    data.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; filename=\"{filename}\"\r\nContent-Type: text/csv\r\n\r\n".encode())
    with open(filename, "rb") as f:
        data.write(f.read())
    data.write(b"\r\n")
add_file("file", "iris_test.csv")
data.write(f"--{boundary}--\r\n".encode())
req_upload = urllib.request.Request(f"{BASE}/upload-dataset", data=data.getvalue(),
    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
up = json.loads(urllib.request.urlopen(req_upload).read())
test("Upload returns filename", up["filename"] == "iris_test.csv")
test("Upload has 30 rows", up["rows"] == 30)
test("Has column names", len(up["column_names"]) == 5)
test("Has describe stats", "mean" in up["describe"]["sepal_length"])
test("Missing values reported", "missing_values" in up)

print("\n=== 6. Training Models ===")
models_to_train = [
    ("classifier", "random_forest", {"file": "iris_test.csv", "form": {"target_column": "species"}}),
    ("classifier", "gradient_boosting", {"file": "iris_test.csv", "form": {"target_column": "species"}}),
    ("classifier", "svm", {"file": "iris_test.csv", "form": {"target_column": "species"}}),
    ("classifier", "logistic_regression", {"file": "iris_test.csv", "form": {"target_column": "species"}}),
    ("regressor", "linear_regression", {"file": "housing_test.csv", "form": {"target_column": "price"}}),
    ("regressor", "ridge", {"file": "housing_test.csv", "form": {"target_column": "price"}}),
    ("regressor", "random_forest_regressor", {"file": "housing_test.csv", "form": {"target_column": "price"}}),
    ("regressor", "gradient_boosting_regressor", {"file": "housing_test.csv", "form": {"target_column": "price"}}),
    ("clustering", "kmeans", {"file": "iris_test.csv", "form": {}}),
    ("clustering", "dbscan", {"file": "iris_test.csv", "form": {}}),
    ("clustering", "hierarchical", {"file": "iris_test.csv", "form": {}}),
    ("neural_network", "neural_network", {"file": "iris_test.csv", "form": {"target_column": "species"}}),
]

for cat, model_id, config in models_to_train:
    try:
        data = io.BytesIO()
        def af(name, value):
            data.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode())
        def aff(name, filename):
            data.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; filename=\"{filename}\"\r\nContent-Type: text/csv\r\n\r\n".encode())
            with open(filename, "rb") as f:
                data.write(f.read())
            data.write(b"\r\n")
        aff("file", config["file"])
        for k, v in config["form"].items():
            af(k, v)
        data.write(f"--{boundary}--\r\n".encode())
        req_t = urllib.request.Request(f"{BASE}/models/{model_id}/train", data=data.getvalue(),
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        try:
            resp = urllib.request.urlopen(req_t)
            result = json.loads(resp.read())
            has_metrics = "metrics" in result and len(result["metrics"]) > 0
            has_training_time = "training_time_seconds" in result
            test(f"Train {model_id}: metrics present", has_metrics, str(result.get("metrics")))
            test(f"Train {model_id}: training time", has_training_time)
            has_model_params = "model_params" in result
            test(f"Train {model_id}: model_params present", has_model_params)
            if "metrics" in result and "accuracy" in result["metrics"]:
                test(f"Train {model_id}: accuracy >= 0.5", result["metrics"]["accuracy"] >= 0.5, str(result["metrics"]["accuracy"]))
            if "metrics" in result and "r2" in result["metrics"]:
                test(f"Train {model_id}: R2 >= 0.9", result["metrics"]["r2"] >= 0.9, str(result["metrics"]["r2"]))
        except urllib.error.HTTPError as e:
            err = e.read().decode()
            test(f"Train {model_id}: HTTP {e.code}", False, err)
    except Exception as e:
        test(f"Train {model_id}: exception", False, str(e))

print("\n=== 7. Predict ===")
# Predict with trained RF (was trained above)
try:
    req_pred = urllib.request.Request(f"{BASE}/predict",
        data=json.dumps({"model_id": "random_forest", "data": [[5.1,3.5,1.4,0.2],[6.3,3.3,6.0,2.5]]}).encode(),
        headers={"Content-Type": "application/json"})
    pred = json.loads(urllib.request.urlopen(req_pred).read())
    test("Predict returns predictions", "predictions" in pred)
    test("Predict returns probabilities", "probabilities" in pred)
    test("Predict returns classes", "classes" in pred)
    test("Predict has 2 results", len(pred["predictions"]) == 2)
except Exception as e:
    test("Predict", False, str(e))

print("\n=== 8. Error Handling ===")
# Non-existent model
try:
    urllib.request.urlopen(f"{BASE}/models/nonexistent")
    test("Nonexistent model GET: should fail", False)
except urllib.error.HTTPError as e:
    test(f"Nonexistent model GET: returns {e.code}", e.code == 404)

# Non-existent model predict
try:
    req_pred = urllib.request.Request(f"{BASE}/predict",
        data=json.dumps({"model_id": "invalid", "data": [[1.0]]}).encode(),
        headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req_pred)
    test("Nonexistent model predict: should fail", False)
except urllib.error.HTTPError as e:
    test(f"Nonexistent model predict: returns {e.code}", e.code == 404)

# Predict without training (create a model that hasn't been used)
# We can't test this easily since models are in memory, but the endpoint has the check
# Actually the models retain state, so we can't test this without restart

print("\n=== 9. L1 Penalty Fix ===")
try:
    boundary = "----L1Boundary"
    data = io.BytesIO()
    data.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"iris_test.csv\"\r\nContent-Type: text/csv\r\n\r\n".encode())
    with open("iris_test.csv", "rb") as f:
        data.write(f.read())
    data.write(b"\r\n")
    data.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"target_column\"\r\n\r\nspecies\r\n".encode())
    data.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"hyperparameters\"\r\n\r\n{{\"penalty\":\"l1\",\"C\":0.5}}\r\n".encode())
    data.write(f"--{boundary}--\r\n".encode())
    req_l1 = urllib.request.Request(f"{BASE}/models/logistic_regression/train", data=data.getvalue(),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    l1 = json.loads(urllib.request.urlopen(req_l1).read())
    test("L1 penalty: accuracy >= 0.5", l1["metrics"]["accuracy"] >= 0.5, str(l1["metrics"]["accuracy"]))
    test("L1 penalty: model_params present", "model_params" in l1)
    actual_penalty = l1.get("model_params", {}).get("penalty", "unknown")
    test(f"L1 penalty: penalty={actual_penalty}", actual_penalty == "l1", actual_penalty)
except urllib.error.HTTPError as e:
    err = e.read().decode()
    test("L1 penalty: HTTP error", False, err)
except Exception as e:
    test("L1 penalty: exception", False, str(e))

print("\n=== 10. DBSCAN with auto-scaling ===")
try:
    boundary = "----DBSCANBoundary"
    data = io.BytesIO()
    data.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"iris_test.csv\"\r\nContent-Type: text/csv\r\n\r\n".encode())
    with open("iris_test.csv", "rb") as f:
        data.write(f.read())
    data.write(b"\r\n")
    data.write(f"--{boundary}--\r\n".encode())
    req_db = urllib.request.Request(f"{BASE}/models/dbscan/train", data=data.getvalue(),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    db = json.loads(urllib.request.urlopen(req_db).read())
    n_clusters = db["metrics"]["n_clusters"]
    n_noise = db["metrics"]["n_noise"]
    test(f"DBSCAN: found {n_clusters} clusters", n_clusters > 1, f"{n_clusters} clusters, {n_noise} noise")
    test(f"DBSCAN: noise <= 10", n_noise <= 10, f"{n_noise} noise points")
except urllib.error.HTTPError as e:
    err = e.read().decode()
    test("DBSCAN: HTTP error", False, err)
except Exception as e:
    test("DBSCAN: exception", False, str(e))

print("\n=== 11. Predict from file ===")
try:
    boundary = "----PFBoundary"
    data = io.BytesIO()
    data.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"iris_test.csv\"\r\nContent-Type: text/csv\r\n\r\n".encode())
    with open("iris_test.csv", "rb") as f:
        data.write(f.read())
    data.write(b"\r\n")
    data.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"target_column\"\r\n\r\nspecies\r\n".encode())
    data.write(f"--{boundary}--\r\n".encode())
    req_pf = urllib.request.Request(f"{BASE}/models/random_forest/predict-from-file", data=data.getvalue(),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    pf = json.loads(urllib.request.urlopen(req_pf).read())
    test("Predict-from-file returns predictions", "predictions" in pf)
    test("Predict-from-file 30 results", len(pf["predictions"]) == 30)
except urllib.error.HTTPError as e:
    err = e.read().decode()
    test("Predict-from-file: HTTP error", False, err)
except Exception as e:
    test("Predict-from-file: exception", False, str(e))

print(f"\n=== RESULTS: {passed} passed, {failed} failed ===")
sys.exit(0 if failed == 0 else 1)
