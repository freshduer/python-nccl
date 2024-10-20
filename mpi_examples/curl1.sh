curl -X POST http://localhost:5000/spawn -H "Content-Type: application/json" \
-d '{
  "program": "python3", 
  "args": ["worker1.py"], 
  "nprocs": 1
}'
