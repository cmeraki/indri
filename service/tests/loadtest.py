import requests
import json
import base64
import numpy as np
import os
from datetime import datetime
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import time
from typing import List, Dict
import torch
import torchaudio

class TTSLoadTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.test_texts = [
            "The quick brown fox jumps over the lazy dog near the riverbank on a sunny afternoon while birds chirp in the distance.",
            "In a bustling city, people hurry along the sidewalks, dodging street vendors and tourists as they make their way to work.",
            "Deep in the ancient forest, towering trees sway gently in the breeze, their leaves creating patterns of light and shadow on the forest floor.",
            "The old lighthouse stands sentinel on the rocky coast, its beam cutting through the thick fog that rolls in from the sea each evening.",
            "Scientists working in the modern laboratory carefully analyze their data, hoping to make a breakthrough in their groundbreaking research project."
        ]
        
        # Create output directories
        self.output_dir = f"debug/load_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.output_dir, exist_ok=True)
        self.audio_dir = os.path.join(self.output_dir, "audio")
        os.makedirs(self.audio_dir, exist_ok=True)
        
        self.results: List[Dict] = []

    def make_request(self, request_id: int):
        time.sleep(0.5)

        text = self.test_texts[request_id % len(self.test_texts)]
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.base_url}/tts",
                json={"text": text, "speaker": "[spkr_jenny_jenny]"},
                headers={"Content-Type": "application/json"}
            )
            end_time = time.time()
            
            if response.status_code == 200:
                response_json = response.json()
                
                # Decode and save audio
                audio_data = base64.b64decode(response_json["array"])
                audio_array = np.frombuffer(audio_data, dtype=np.float32)
                audio_array = torch.from_numpy(audio_array)
                audio_filename = f"audio_{request_id}.wav"
                audio_path = os.path.join(self.audio_dir, audio_filename)
                
                result = {
                    "request_id": request_id,
                    "text": text,
                    "audio_path": audio_path,
                    "audio_array": audio_array,
                    "response_time": end_time - start_time,
                    "status": response.status_code,
                    "time_to_first_token": response_json["metrics"]["time_to_first_token"],
                    "time_to_last_token": response_json["metrics"]["time_to_last_token"],
                    "generate_end_to_end_time": response_json["metrics"]["generate_end_to_end_time"]
                }
                print(f"Request {request_id} completed in {result['response_time']:.2f}s")
                
            else:
                result = {
                    "request_id": request_id,
                    "text": text,
                    "error": f"HTTP {response.status_code}",
                    "status": response.status_code,
                    "response_time": end_time - start_time
                }
                print(f"Request {request_id} failed with status {response.status_code}")
                
            self.results.append(result)
            
        except Exception as e:
            print(f"Error in request {request_id}: {str(e)}")
            self.results.append({
                "request_id": request_id,
                "text": text,
                "error": str(e),
                "status": "failed",
                "response_time": time.time() - start_time
            })

    def run_load_test(self, num_concurrent: int = 5):
        print(f"Starting load test with {num_concurrent} concurrent requests...")

        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            executor.map(self.make_request, range(num_concurrent))

        self.save_results()
        # self.analyze_results()

    def save_results(self):
        # Save test texts
        with open(os.path.join(self.output_dir, "test_texts.json"), "w") as f:
            json.dump(self.test_texts, f, indent=2)

        for r in self.results:
            torchaudio.save(r["audio_path"], r["audio_array"].unsqueeze(0), sample_rate=24000)

        # Save results as CSV
        df = pd.DataFrame(self.results)
        df.to_csv(os.path.join(self.output_dir, "results.csv"), index=False)
        
        print(f"\nResults saved to {self.output_dir}/")

    def analyze_results(self):
        df = pd.DataFrame(self.results)
        successful_requests = df[df['status'] == 200]
        
        print("\nTest Results Summary:")
        print("-" * 50)
        print(f"Total requests: {len(df)}")
        print(f"Successful requests: {len(successful_requests)}")
        print(f"Failed requests: {len(df) - len(successful_requests)}")
        
        if not successful_requests.empty:
            print(f"\nResponse Time Statistics (seconds):")
            print(f"Average: {successful_requests['response_time'].mean():.2f}")
            print(f"Min: {successful_requests['response_time'].min():.2f}")
            print(f"Max: {successful_requests['response_time'].max():.2f}")
            print(f"Median: {successful_requests['response_time'].median():.2f}")
            
            print(f"\nGeneration Time Statistics (seconds):")
            print(f"Average time to first token: {successful_requests['time_to_first_token'].mean():.2f}")
            print(f"Average time to last token: {successful_requests['time_to_last_token'].mean():.2f}")
            print(f"Average end-to-end time: {successful_requests['generate_end_to_end_time'].mean():.2f}")

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--n", type=int, default=1)
    args = parser.parse_args()

    tester = TTSLoadTester()
    tester.run_load_test(num_concurrent=args.n)  # Adjust number of concurrent requests here
