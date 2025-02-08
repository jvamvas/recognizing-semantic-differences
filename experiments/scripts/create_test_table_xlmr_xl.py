from collections import OrderedDict

from transformers import pipeline

from experiments.benchmark import DifferenceRecognitionResult
from experiments.utils import load_summary_benchmarks
from recognizers import DiffAlign


benchmarks = load_summary_benchmarks("test")
device = 0

recognizers = []
recognizers.append(DiffAlign(
    pipeline=pipeline(
        model="facebook/xlm-roberta-xl",
        task="feature-extraction",
    ),
    batch_size=4,
))
recognizers.append(DiffAlign(
    pipeline=pipeline(
        model="facebook/xlm-roberta-xxl",
        task="feature-extraction",
    ),
    batch_size=1,
))

results = OrderedDict()
for i, recognizer in enumerate(recognizers):
    print(recognizer)
    recognizer.pipeline.device = device
    recognizer.device = device
    recognizer.pipeline.model = recognizer.pipeline.model.to(device)
    recognizer_results = []
    for benchmark in benchmarks:
        print(benchmark)
        result = benchmark.evaluate(recognizer)
        print(result)
        recognizer_results.append(result)
    # Average last six results (cross-lingual)
    recognizer_results, cross_lingual_results = recognizer_results[:-6], recognizer_results[-6:]
    cross_lingual_mean = sum([result.spearman for result in cross_lingual_results]) / len(cross_lingual_results)
    recognizer_results.append(DifferenceRecognitionResult(spearman=cross_lingual_mean))
    results[str(recognizer)] = recognizer_results
    recognizers[i] = None
    del recognizer

print(results)
