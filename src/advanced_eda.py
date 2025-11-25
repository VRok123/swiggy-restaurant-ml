import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud
from utils import logger, setup_plotting
from config import *

class AdvancedEDA:
    def __init__(self, dishes_df: pd.DataFrame, restaurants_df: pd.DataFrame):
        self.dishes_df = dishes_df
        self.restaurants_df = restaurants_df
        setup_plotting()
    
    def analyze_dish_level_data(self):
        """Comprehensive analysis of dish-level data"""
        print("🍽️ Dish-Level Analysis")
        print("=" * 50)
        
        # Price distribution by category
        self.plot_price_by_category()
        
        # Rating distribution
        self.plot_rating_analysis()
        
        # Popular dishes analysis
        self.analyze_popular_dishes()
        
        # Geographic analysis
        self.analyze_geographic_distribution()
    
    def plot_price_by_category(self):
        """Plot price distribution by food category"""
        plt.figure(figsize=(15, 8))
        
        # Get top categories by count
        top_categories = self.dishes_df['category'].value_counts().head(15).index
        
        filtered_df = self.dishes_df[self.dishes_df['category'].isin(top_categories)]
        
        # Box plot of prices by category
        sns.boxplot(data=filtered_df, x='category', y='price')
        plt.title('Price Distribution by Food Category', fontweight='bold', fontsize=14)
        plt.xlabel('Food Category')
        plt.ylabel('Price (INR)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'price_by_category.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Average price by category
        avg_price_by_category = (filtered_df.groupby('category')['price']
                                .mean()
                                .sort_values(ascending=False)
                                .head(15))
        
        plt.figure(figsize=(12, 6))
        avg_price_by_category.plot(kind='bar', color='lightcoral', alpha=0.7)
        plt.title('Average Price by Food Category (Top 15)', fontweight='bold')
        plt.xlabel('Food Category')
        plt.ylabel('Average Price (INR)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'avg_price_by_category.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_rating_analysis(self):
        """Analyze rating distributions and patterns"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Rating distribution
        self.dishes_df['rating'].hist(bins=30, ax=axes[0,0], alpha=0.7, color='skyblue')
        axes[0,0].set_title('Distribution of Dish Ratings', fontweight='bold')
        axes[0,0].set_xlabel('Rating')
        axes[0,0].set_ylabel('Frequency')
        
        # Rating vs Price scatter
        axes[0,1].scatter(self.dishes_df['rating'], self.dishes_df['price'], 
                         alpha=0.3, color='green')
        axes[0,1].set_title('Rating vs Price', fontweight='bold')
        axes[0,1].set_xlabel('Rating')
        axes[0,1].set_ylabel('Price (INR)')
        
        # Rating count distribution (log scale)
        self.dishes_df['rating_count'].hist(bins=50, ax=axes[1,0], alpha=0.7, color='orange')
        axes[1,0].set_title('Distribution of Rating Counts', fontweight='bold')
        axes[1,0].set_xlabel('Rating Count')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].set_yscale('log')
        
        # Price vs Rating count
        axes[1,1].scatter(self.dishes_df['price'], self.dishes_df['rating_count'],
                         alpha=0.3, color='purple')
        axes[1,1].set_title('Price vs Rating Count', fontweight='bold')
        axes[1,1].set_xlabel('Price (INR)')
        axes[1,1].set_ylabel('Rating Count')
        axes[1,1].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'rating_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_popular_dishes(self):
        """Analyze most popular dishes"""
        # Dishes with highest rating counts
        popular_dishes = (self.dishes_df
                         .nlargest(20, 'rating_count')
                         [['dish_name', 'category', 'rating', 'rating_count', 'price']])
        
        print("\n🏆 Top 20 Most Rated Dishes:")
        print(popular_dishes.to_string(index=False))
        
        # Highest rated dishes (with minimum rating count)
        min_ratings = 50
        highly_rated = (self.dishes_df[self.dishes_df['rating_count'] >= min_ratings]
                       .nlargest(15, 'rating')
                       [['dish_name', 'category', 'rating', 'rating_count', 'price']])
        
        print(f"\n⭐ Top 15 Highest Rated Dishes (min {min_ratings} ratings):")
        print(highly_rated.to_string(index=False))
    
    def analyze_geographic_distribution(self):
        """Analyze geographic distribution of restaurants"""
        # Restaurants by city
        city_counts = self.restaurants_df['city'].value_counts().head(15)
        
        plt.figure(figsize=(12, 6))
        city_counts.plot(kind='bar', color='lightseagreen', alpha=0.7)
        plt.title('Restaurant Distribution by City (Top 15)', fontweight='bold')
        plt.xlabel('City')
        plt.ylabel('Number of Restaurants')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'restaurants_by_city.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Average rating by city
        avg_rating_by_city = (self.restaurants_df.groupby('city')['avg_rating']
                             .mean()
                             .sort_values(ascending=False)
                             .head(15))
        
        plt.figure(figsize=(12, 6))
        avg_rating_by_city.plot(kind='bar', color='goldenrod', alpha=0.7)
        plt.title('Average Restaurant Rating by City (Top 15)', fontweight='bold')
        plt.xlabel('City')
        plt.ylabel('Average Rating')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'avg_rating_by_city.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_restaurant_level_data(self):
        """Analyze restaurant-level aggregated data"""
        print("\n🏪 Restaurant-Level Analysis")
        print("=" * 50)
        
        # Restaurant size distribution
        self.plot_restaurant_size_distribution()
        
        # Price vs Rating analysis
        self.plot_price_vs_rating()
        
        # Target variable analysis
        self.analyze_target_variables()
    
    def plot_restaurant_size_distribution(self):
        """Plot distribution of restaurant sizes (number of dishes)"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Dish count distribution
        self.restaurants_df['dish_count'].hist(bins=50, ax=axes[0], alpha=0.7, color='teal')
        axes[0].set_title('Distribution of Dishes per Restaurant', fontweight='bold')
        axes[0].set_xlabel('Number of Dishes')
        axes[0].set_ylabel('Frequency')
        
        # Restaurant price distribution
        self.restaurants_df['avg_price'].hist(bins=50, ax=axes[1], alpha=0.7, color='coral')
        axes[1].set_title('Distribution of Average Restaurant Prices', fontweight='bold')
        axes[1].set_xlabel('Average Price (INR)')
        axes[1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'restaurant_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_price_vs_rating(self):
        """Plot relationship between price and rating"""
        plt.figure(figsize=(10, 6))
        
        plt.scatter(self.restaurants_df['avg_price'], self.restaurants_df['avg_rating'],
                   alpha=0.5, c=self.restaurants_df['total_rating_count'], 
                   cmap='viridis', s=20)
        
        plt.colorbar(label='Total Rating Count')
        plt.title('Restaurant: Average Price vs Average Rating', fontweight='bold')
        plt.xlabel('Average Price (INR)')
        plt.ylabel('Average Rating')
        
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'price_vs_rating.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_target_variables(self):
        """Analyze the target variables for classification"""
        target_columns = ['is_high_rated', 'is_popular', 'is_premium']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, target in enumerate(target_columns):
            if target in self.restaurants_df.columns:
                value_counts = self.restaurants_df[target].value_counts()
                axes[i].pie(value_counts.values, labels=value_counts.index, 
                           autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightcoral'])
                axes[i].set_title(f'Distribution of {target}', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'target_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print target statistics
        print("\n🎯 Target Variable Statistics:")
        for target in target_columns:
            if target in self.restaurants_df.columns:
                counts = self.restaurants_df[target].value_counts()
                print(f"   {target}: {counts[1]} positive, {counts[0]} negative "
                      f"({counts[1]/len(self.restaurants_df)*100:.1f}% positive)")

def run_advanced_eda(dishes_df: pd.DataFrame, restaurants_df: pd.DataFrame):
    """Run comprehensive EDA on both dish and restaurant level data"""
    eda = AdvancedEDA(dishes_df, restaurants_df)
    
    # Dish-level analysis
    eda.analyze_dish_level_data()
    
    # Restaurant-level analysis  
    eda.analyze_restaurant_level_data()
    
    logger.info("✅ Advanced EDA completed!")