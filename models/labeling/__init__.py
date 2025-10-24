"""
Labeling module - Triple-barrier and meta-labeling
"""
from models.labeling.triple_barrier import TripleBarrierLabeling, apply_triple_barrier_labeling
from models.labeling.meta_labeling import MetaLabeling, create_meta_labeling_pipeline
from models.labeling.sample_weights import SampleWeights
from models.labeling.event_sampling import EventSampling, detect_events

__all__ = [
    'TripleBarrierLabeling',
    'apply_triple_barrier_labeling',
    'MetaLabeling',
    'create_meta_labeling_pipeline',
    'SampleWeights',
    'EventSampling',
    'detect_events'
]
