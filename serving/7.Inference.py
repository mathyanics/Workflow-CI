"""Local JSON inference API with Prometheus counters, gauges and histograms."""
import json, logging, os, time
from http.server import BaseHTTPRequestHandler, HTTPServer
import joblib
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST

MODEL_PATH=os.getenv('MODEL_PATH','../Membangun_model/model_artifacts/credit_default_model.joblib')
model=joblib.load(MODEL_PATH)
REQUESTS=Counter('credit_api_requests_total','API request count',['route','status'])
REQUEST_DURATION=Histogram('credit_api_request_duration_seconds','End-to-end request latency',['route'])
IN_PROGRESS=Gauge('credit_api_requests_in_progress','Requests currently being processed')
PAYLOAD_BYTES=Histogram('credit_api_request_payload_bytes','Incoming prediction payload size in bytes',buckets=[100,500,1000,5000,10000,50000])
INVALID=Counter('credit_api_invalid_requests_total','Invalid prediction requests')
INFERENCE_DURATION=Histogram('credit_api_inference_duration_seconds','Model inference time in seconds',buckets=[0.001,0.005,0.01,0.05,0.1,0.5,1.0])
PREDICTED=Counter('credit_api_predictions_total','Predicted class count',['prediction'])
PROBABILITY=Histogram('credit_api_prediction_probability','Predicted default probability',buckets=[0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.99,1.0])
BATCH_SIZE=Histogram('credit_api_prediction_batch_size','Records per prediction request',buckets=[1,2,5,10,25,50,100])
POSITIVE_RATIO=Gauge('credit_api_positive_prediction_ratio','Positive fraction in the latest request')
UP=Gauge('credit_api_up','API process up')
UP.set(1)
logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path=='/health':
            REQUESTS.labels('health','200').inc()
            self.send_response(200); self.send_header('Content-Type','application/json'); self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
        elif self.path=='/metrics':
            body=generate_latest(); self.send_response(200); self.send_header('Content-Type',CONTENT_TYPE_LATEST); self.end_headers(); self.wfile.write(body)
        else: self.send_error(404)

    def do_POST(self):
        if self.path!='/predict': self.send_error(404); return
        started=time.perf_counter(); IN_PROGRESS.inc()
        try:
            size=int(self.headers.get('Content-Length','0'))
            PAYLOAD_BYTES.observe(size)
            payload=json.loads(self.rfile.read(size))
            records=payload if isinstance(payload,list) else [payload]
            import pandas as pd
            X=pd.DataFrame(records,columns=['student','balance','income'])
            X['student']=X['student'].astype(str).str.strip().str.lower()
            infer_started=time.perf_counter()
            preds=model.predict(X).tolist(); probs=model.predict_proba(X)[:,1].tolist()
            INFERENCE_DURATION.observe(time.perf_counter()-infer_started)
            for pred in preds: PREDICTED.labels(str(int(pred))).inc()
            for prob in probs: PROBABILITY.observe(float(prob))
            BATCH_SIZE.observe(len(preds)); POSITIVE_RATIO.set(sum(preds)/len(preds) if preds else 0)
            results=[{'default_prediction':int(pred),'default_probability':float(prob)} for pred,prob in zip(preds,probs)]
            body=json.dumps(results if isinstance(payload,list) else results[0]).encode()
            self.send_response(200); self.send_header('Content-Type','application/json'); self.end_headers(); self.wfile.write(body)
            REQUESTS.labels('predict','200').inc(); logging.info('prediction_count=%d',len(preds))
        except Exception:
            INVALID.inc(); REQUESTS.labels('predict','400').inc(); logging.exception('invalid prediction request'); self.send_error(400,'Invalid prediction payload')
        finally:
            REQUEST_DURATION.labels('predict').observe(time.perf_counter()-started); IN_PROGRESS.dec()

    def log_message(self,fmt,*args): logging.info(fmt,*args)

if __name__=='__main__':
    host=os.getenv('HOST','127.0.0.1'); port=int(os.getenv('PORT','8000'))
    logging.info('serving on %s:%s',host,port); HTTPServer((host,port),Handler).serve_forever()
