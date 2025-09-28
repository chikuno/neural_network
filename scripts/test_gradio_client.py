from gradio_client import Client

BASE = "http://127.0.0.1:7860/"
client = Client(BASE)

print("Calling /run ...", flush=True)
try:
    r1 = client.predict(
        prompt="Hello!!",
        num_tokens=40,
        temperature=0.9,
        decoding="sampling",
        top_k=30,
        top_p=0.95,
        model_sel_v="Auto",
        stream=False,
        api_name="/run",
    )
    s = str(r1)
    print("RUN RESULT:", s[:300], flush=True)
except Exception as e:
    print("RUN ERROR:", repr(e), flush=True)

print("\nCalling /run_stream ...", flush=True)
try:
    r2 = client.predict(
        prompt="Hello!!",
        num_tokens=40,
        temperature=0.9,
        decoding="sampling",
        top_k=30,
        top_p=0.95,
        model_sel_v="Auto",
        api_name="/run_stream",
    )
    s2 = str(r2)
    print("STREAM RESULT:", s2[:300], flush=True)
except Exception as e:
    print("STREAM ERROR:", repr(e), flush=True)
