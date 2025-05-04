"""
Cat identification module
"""
from .cat_profiles import CatProfile, CatProfileManager
from .feature_extractor import FeatureExtractor

__all__ = ['CatProfile', 'CatProfileManager', 'FeatureExtractor']