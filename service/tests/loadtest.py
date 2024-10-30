import os
import time
import json
import torch
import asyncio
import aiohttp
import base64
import random
import numpy as np
import pandas as pd
import torchaudio
from datetime import datetime, timedelta

from typing import List, Dict, Optional
from dataclasses import dataclass

from ..models import TTSResponse
from ..logger import get_logger

logger = get_logger(__name__)

@dataclass
class RequestResult:
    request_id: int
    text: str
    start_time: float
    end_time: float
    response_time: float
    status: int
    error: Optional[str] = None
    audio_path: Optional[str] = None
    metrics: Optional[Dict] = None
    timestamp: Optional[datetime] = None

class TTSLoadTester:
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        test_duration_minutes: int = 60,
        min_qps: float = 0.5,
        max_qps: float = 5.0,
        qps_step: float = 0.5,
        step_duration_minutes: int = 5
    ):
        self.base_url = base_url
        self.test_duration = timedelta(minutes=test_duration_minutes)
        self.min_qps = min_qps
        self.max_qps = max_qps
        self.qps_step = qps_step
        self.step_duration = timedelta(minutes=step_duration_minutes)

        with open('service/tests/test_data.txt', 'r', encoding='utf-8') as f:
            self.test_texts = [line.strip() for line in f if line.strip()]

        logger.info(f"Loaded {len(self.test_texts)} test texts")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = f"debug/load_test_results_{timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)

        self.audio_dir = os.path.join(self.output_dir, "audio")
        os.makedirs(self.audio_dir, exist_ok=True)

        self.results: List[RequestResult] = []
        self.current_qps = min_qps

    async def make_request(self, session: aiohttp.ClientSession):
        text = random.choice(self.test_texts)

        start_time = time.time()
        timestamp = datetime.now()

        try:
            async with session.post(
                f"{self.base_url}/tts",
                json={"text": text, "speaker": "[spkr_jenny_jenny]"},
                headers={"Content-Type": "application/json"}
            ) as response:
                end_time = time.time()
                response_time = end_time - start_time
                
                if response.status == 200:
                    response_data = await response.json()
                    response_obj = TTSResponse(**response_data)

                    request_id = response_obj.request_id

                    audio_data = base64.b64decode(response_obj.array)
                    audio_array = np.frombuffer(audio_data, dtype=np.float32)
                    audio_array = torch.from_numpy(audio_array.copy())
                    audio_filename = f"{request_id}.wav"
                    audio_path = os.path.join(self.audio_dir, audio_filename)

                    await asyncio.to_thread(
                        torchaudio.save,
                        audio_path,
                        audio_array.unsqueeze(0),
                        sample_rate=response_obj.sample_rate
                    )
                    
                    result = RequestResult(
                        request_id=request_id,
                        text=text,
                        start_time=start_time,
                        end_time=end_time,
                        response_time=response_time,
                        status=response.status,
                        audio_path=audio_path,
                        metrics=response_obj.metrics.model_dump() if response_obj.metrics else None,
                        timestamp=timestamp
                    )
                    # logger.info(f"Request {request_id} completed in {response_time:.2f}s")

                else:
                    result = RequestResult(
                        request_id=None,
                        text=text,
                        start_time=start_time,
                        end_time=end_time,
                        response_time=response_time,
                        status=response.status,
                        error=f"HTTP {response.status}",
                        timestamp=timestamp
                    )
                    logger.warning(f"Request failed with status {response.status}")

        except Exception as e:
            error_time = time.time()
            logger.error(f"Error in request {request_id}: {str(e)}")
            result = RequestResult(
                request_id=None,
                text=text,
                start_time=start_time,
                end_time=error_time,
                response_time=error_time - start_time,
                status=-1,
                error=str(e),
                timestamp=timestamp
            )

        self.results.append(result)

    async def request_generator(self):
        start_time = datetime.now()
        last_qps_change = start_time
        current_qps = self.min_qps

        async with aiohttp.ClientSession() as session:
            while datetime.now() - start_time < self.test_duration:
                current_time = datetime.now()
                
                # Update QPS if step duration has elapsed
                if current_time - last_qps_change >= self.step_duration:
                    current_qps = min(current_qps + self.qps_step, self.max_qps)
                    last_qps_change = current_time
                    logger.info(f"Increasing QPS to {current_qps}")

                # Calculate delay based on current QPS
                delay = 1.0 / current_qps
                asyncio.create_task(self.make_request(session))
                await asyncio.sleep(delay)

    async def run_load_test(self):
        logger.info(f"Starting load test for {self.test_duration.total_seconds() / 60:.1f} minutes")
        logger.info(f"QPS will increase from {self.min_qps} to {self.max_qps} " 
                    f"every {self.step_duration.total_seconds() / 60:.1f} minutes")

        await self.request_generator()

        self.save_results()
        self.analyze_results()

    def save_results(self):
        # Save test configuration
        config = {
            "base_url": self.base_url,
            "test_duration_minutes": self.test_duration.total_seconds() / 60,
            "min_qps": self.min_qps,
            "max_qps": self.max_qps,
            "qps_step": self.qps_step,
            "step_duration_minutes": self.step_duration.total_seconds() / 60,
            "test_texts": self.test_texts
        }
        
        with open(os.path.join(self.output_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        df = pd.DataFrame([vars(r) for r in self.results])

        if not df.empty:
            df['relative_time'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds()

        df.to_csv(os.path.join(self.output_dir, "detailed_results.csv"), index=False)

        def q95(x):
            return x.quantile(0.95)

        time_series = df.set_index('timestamp').resample('1s').agg({
            'request_id': 'count',
            'response_time': ['mean', 'min', 'max', 'std', q95],
            'status': lambda x: (x == 200).mean() * 100  # success rate
        }).reset_index()

        time_series.columns = ['timestamp', 'requests_per_s', 'avg_response_time',
                             'min_response_time', 'max_response_time',
                             'std_response_time', 'p95_response_time', 'success_rate']

        time_series.to_csv(os.path.join(self.output_dir, "time_series_metrics.csv"), index=False)

        logger.info(f"Results saved to {self.output_dir}/")

    def analyze_results(self):
        df = pd.DataFrame([vars(r) for r in self.results])
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
            
            if 'metrics' in successful_requests.columns:
                metrics_df = pd.DataFrame(list(successful_requests['metrics'].dropna()))
                if not metrics_df.empty:
                    print(f"\nGeneration Time Statistics (seconds):")
                    print(f"Average time to first token: {metrics_df['time_to_first_token'].apply(lambda x: x[0]).mean():.2f}")
                    print(f"Average time to last token: {metrics_df['time_to_last_token'].apply(lambda x: x[0]).mean():.2f}")
                    print(f"Average end-to-end time: {metrics_df['generate_end_to_end_time'].mean():.2f}")

if __name__ == "__main__":
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument("--duration", type=int, default=60, help="Test duration in minutes")
    parser.add_argument("--min-qps", type=float, default=0.5, help="Starting queries per second")
    parser.add_argument("--max-qps", type=float, default=5.0, help="Maximum queries per second")
    parser.add_argument("--qps-step", type=float, default=0.5, help="QPS increase step")
    parser.add_argument("--url", type=str, default="http://localhost:8000", help="Base URL fo[r the TTS service")

    args = parser.parse_args()

    step_duration = args.duration / (args.max_qps / args.qps_step)

    tester = TTSLoadTester(
        base_url=args.url,
        test_duration_minutes=args.duration,
        min_qps=args.min_qps,
        max_qps=args.max_qps,
        qps_step=args.qps_step,
        step_duration_minutes=step_duration
    )

    asyncio.run(tester.run_load_test())
