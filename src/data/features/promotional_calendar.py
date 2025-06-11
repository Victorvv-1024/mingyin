"""
Chinese E-commerce Promotional Calendar

This module implements comprehensive knowledge of Chinese e-commerce promotional periods,
including major shopping festivals, seasonal campaigns, and platform-specific events.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class ChineseEcommerceCalendar:
    """Comprehensive Chinese e-commerce promotional calendar."""
    
    def __init__(self):
        """Initialize the promotional calendar with major Chinese e-commerce events."""
        self.promotional_events = self._initialize_promotional_events()
        self.seasonal_patterns = self._initialize_seasonal_patterns()
        logger.info("Chinese E-commerce Calendar initialized")
    
    def _initialize_promotional_events(self) -> Dict[str, Dict]:
        """Initialize major Chinese e-commerce promotional events."""
        
        # Major recurring promotional events
        events = {
            "Singles Day": {
                "dates": [(11, 11)],  # November 11
                "duration_days": 3,   # Pre and post promotion
                "intensity": 1.0,     # Maximum intensity
                "description": "Largest Chinese shopping festival"
            },
            "Chinese New Year": {
                "dates": [(1, 20), (1, 25), (2, 1), (2, 5), (2, 10), (2, 15)],  # Varies by year
                "duration_days": 15,
                "intensity": 0.9,
                "description": "Spring Festival shopping period"
            },
            "618 Shopping Festival": {
                "dates": [(6, 18)],  # June 18 (JD.com anniversary)
                "duration_days": 7,
                "intensity": 0.8,
                "description": "Mid-year shopping festival"
            },
            "Double 12": {
                "dates": [(12, 12)],  # December 12
                "duration_days": 2,
                "intensity": 0.7,
                "description": "Year-end shopping festival"
            },
            "May Day Golden Week": {
                "dates": [(5, 1)],
                "duration_days": 7,
                "intensity": 0.6,
                "description": "Labor Day holiday shopping"
            },
            "National Day Golden Week": {
                "dates": [(10, 1)],
                "duration_days": 7,
                "intensity": 0.6,
                "description": "National Day holiday shopping"
            },
            "Qixi Festival": {
                "dates": [(8, 14), (8, 20), (8, 25)],  # Varies by lunar calendar
                "duration_days": 3,
                "intensity": 0.5,
                "description": "Chinese Valentine's Day"
            },
            "Mother's Day": {
                "dates": [(5, 8), (5, 15)],  # Second Sunday in May (approximate)
                "duration_days": 3,
                "intensity": 0.4,
                "description": "Mother's Day promotions"
            },
            "Father's Day": {
                "dates": [(6, 15), (6, 22)],  # Third Sunday in June (approximate)
                "duration_days": 3,
                "intensity": 0.4,
                "description": "Father's Day promotions"
            },
            "Back to School": {
                "dates": [(8, 15), (9, 1)],
                "duration_days": 14,
                "intensity": 0.5,
                "description": "Back to school shopping season"
            }
        }
        
        return events
    
    def _initialize_seasonal_patterns(self) -> Dict[str, Dict]:
        """Initialize seasonal shopping patterns."""
        
        patterns = {
            "Winter Holiday Season": {
                "months": [12, 1, 2],
                "intensity": 0.8,
                "description": "Winter holidays and Chinese New Year period"
            },
            "Spring Season": {
                "months": [3, 4, 5],
                "intensity": 0.6,
                "description": "Spring shopping and holiday preparation"
            },
            "Summer Season": {
                "months": [6, 7, 8],
                "intensity": 0.7,
                "description": "Summer shopping with 618 and back-to-school"
            },
            "Autumn Season": {
                "months": [9, 10, 11],
                "intensity": 0.9,
                "description": "Peak shopping season with Singles Day"
            }
        }
        
        return patterns
    
    def is_promotional_period(self, date: pd.Timestamp) -> bool:
        """
        Check if a given date falls within any promotional period.
        
        Args:
            date: Date to check
            
        Returns:
            True if date is within a promotional period
        """
        month = date.month
        day = date.day
        
        for event_name, event_info in self.promotional_events.items():
            for promo_month, promo_day in event_info["dates"]:
                # Check if date falls within promotion duration
                promo_date = datetime(date.year, promo_month, promo_day)
                duration = timedelta(days=event_info["duration_days"])
                
                if promo_date - duration/2 <= date <= promo_date + duration/2:
                    return True
        
        return False
    
    def is_specific_event(self, date: pd.Timestamp, event_name: str) -> bool:
        """
        Check if a date falls within a specific promotional event.
        
        Args:
            date: Date to check
            event_name: Name of the promotional event
            
        Returns:
            True if date is within the specified event period
        """
        if event_name not in self.promotional_events:
            return False
        
        event_info = self.promotional_events[event_name]
        month = date.month
        day = date.day
        
        for promo_month, promo_day in event_info["dates"]:
            # Check if date falls within promotion duration
            promo_date = datetime(date.year, promo_month, promo_day)
            duration = timedelta(days=event_info["duration_days"])
            
            if promo_date - duration/2 <= date <= promo_date + duration/2:
                return True
        
        return False
    
    def get_promotional_intensity(self, date: pd.Timestamp) -> float:
        """
        Get the promotional intensity for a given date.
        
        Args:
            date: Date to check
            
        Returns:
            Promotional intensity score (0.0 to 1.0)
        """
        max_intensity = 0.0
        month = date.month
        day = date.day
        
        # Check promotional events
        for event_name, event_info in self.promotional_events.items():
            for promo_month, promo_day in event_info["dates"]:
                promo_date = datetime(date.year, promo_month, promo_day)
                duration = timedelta(days=event_info["duration_days"])
                
                if promo_date - duration/2 <= date <= promo_date + duration/2:
                    # Calculate intensity based on distance from peak date
                    days_from_peak = abs((date - promo_date).days)
                    decay_factor = max(0, 1 - (days_from_peak / (event_info["duration_days"]/2)))
                    intensity = event_info["intensity"] * decay_factor
                    max_intensity = max(max_intensity, intensity)
        
        # Add seasonal baseline intensity
        for season_name, season_info in self.seasonal_patterns.items():
            if month in season_info["months"]:
                seasonal_intensity = season_info["intensity"] * 0.3  # Reduced baseline
                max_intensity = max(max_intensity, seasonal_intensity)
        
        return min(max_intensity, 1.0)
    
    def days_to_next_promotion(self, date: pd.Timestamp) -> int:
        """
        Calculate days until the next promotional period.
        
        Args:
            date: Current date
            
        Returns:
            Number of days to next promotion (max 365)
        """
        min_days = 365
        current_year = date.year
        
        # Check events in current year
        for event_name, event_info in self.promotional_events.items():
            for promo_month, promo_day in event_info["dates"]:
                promo_date = datetime(current_year, promo_month, promo_day)
                
                if promo_date > date:
                    days_diff = (promo_date - date).days
                    min_days = min(min_days, days_diff)
        
        # Check events in next year if no upcoming events in current year
        if min_days == 365:
            for event_name, event_info in self.promotional_events.items():
                for promo_month, promo_day in event_info["dates"]:
                    promo_date = datetime(current_year + 1, promo_month, promo_day)
                    days_diff = (promo_date - date).days
                    min_days = min(min_days, days_diff)
        
        return min(min_days, 365)
    
    def days_from_last_promotion(self, date: pd.Timestamp) -> int:
        """
        Calculate days since the last promotional period.
        
        Args:
            date: Current date
            
        Returns:
            Number of days since last promotion (max 365)
        """
        max_days = 0
        current_year = date.year
        
        # Check events in current year
        for event_name, event_info in self.promotional_events.items():
            for promo_month, promo_day in event_info["dates"]:
                promo_date = datetime(current_year, promo_month, promo_day)
                
                if promo_date < date:
                    days_diff = (date - promo_date).days
                    max_days = max(max_days, days_diff)
        
        # Check events in previous year if no past events in current year
        if max_days == 0 and date.month <= 6:  # Only check previous year for early months
            for event_name, event_info in self.promotional_events.items():
                for promo_month, promo_day in event_info["dates"]:
                    promo_date = datetime(current_year - 1, promo_month, promo_day)
                    days_diff = (date - promo_date).days
                    if days_diff > 0:
                        max_days = max(max_days, days_diff)
        
        return min(max_days, 365)
    
    def get_promotional_events(self) -> Dict[str, Dict]:
        """Get all promotional events."""
        return self.promotional_events
    
    def get_seasonal_patterns(self) -> Dict[str, Dict]:
        """Get all seasonal patterns."""
        return self.seasonal_patterns
    
    def get_month_promotional_score(self, month: int) -> float:
        """
        Get the general promotional intensity for a given month.
        
        Args:
            month: Month number (1-12)
            
        Returns:
            Monthly promotional score (0.0 to 1.0)
        """
        monthly_scores = {
            1: 0.8,   # Chinese New Year
            2: 0.7,   # Chinese New Year continuation
            3: 0.4,   # Spring season
            4: 0.4,   # Spring season
            5: 0.6,   # May Day Golden Week
            6: 0.8,   # 618 Shopping Festival
            7: 0.5,   # Summer season
            8: 0.6,   # Back to school, Qixi
            9: 0.5,   # Autumn preparation
            10: 0.7,  # National Day, Singles Day preparation
            11: 1.0,  # Singles Day - Peak shopping month
            12: 0.8   # Double 12, Year-end shopping
        }
        
        return monthly_scores.get(month, 0.4)
    
    def get_promotional_calendar_features(self, date: pd.Timestamp) -> Dict[str, float]:
        """
        Get comprehensive promotional calendar features for a given date.
        
        Args:
            date: Date to analyze
            
        Returns:
            Dictionary of promotional features
        """
        features = {}
        
        # Basic promotional indicators
        features['is_promotional_period'] = float(self.is_promotional_period(date))
        features['promotional_intensity'] = self.get_promotional_intensity(date)
        features['days_to_next_promo'] = float(self.days_to_next_promotion(date))
        features['days_from_last_promo'] = float(self.days_from_last_promotion(date))
        features['monthly_promo_score'] = self.get_month_promotional_score(date.month)
        
        # Specific event indicators
        for event_name in self.promotional_events.keys():
            safe_name = event_name.lower().replace(" ", "_").replace("'", "")
            features[f'is_{safe_name}'] = float(self.is_specific_event(date, event_name))
        
        # Seasonal patterns
        for season_name, season_info in self.seasonal_patterns.items():
            safe_name = season_name.lower().replace(" ", "_")
            features[f'is_{safe_name}'] = float(date.month in season_info['months'])
        
        return features
    
    def analyze_promotional_trends(self, df: pd.DataFrame, date_col: str = 'sales_month') -> pd.DataFrame:
        """
        Analyze promotional trends in sales data.
        
        Args:
            df: DataFrame with sales data
            date_col: Name of the date column
            
        Returns:
            DataFrame with promotional trend analysis
        """
        analysis_df = df.copy()
        analysis_df[date_col] = pd.to_datetime(analysis_df[date_col])
        
        # Add promotional features
        for _, row in analysis_df.iterrows():
            date = row[date_col]
            promo_features = self.get_promotional_calendar_features(date)
            
            for feature_name, feature_value in promo_features.items():
                if feature_name not in analysis_df.columns:
                    analysis_df[feature_name] = 0.0
                analysis_df.loc[_, feature_name] = feature_value
        
        return analysis_df