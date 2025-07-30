#!/usr/bin/env python3
"""
Automation Demo Script
Demonstrates the complete automated workflow:
1. Image scraping and classification
2. Automated analysis and visualization
3. Dashboard generation
"""

import os
import sys
import time
import requests
from pathlib import Path
import logging

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.dashboard_system import DashboardSystem
from app.image_analyzer import ImageAnalyzer
from app.enhanced_classifier import EnhancedImageClassifier

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AutomationDemo")

class AutomationDemo:
    """
    Demo class to showcase the automated workflow
    """
    
    def __init__(self):
        self.dashboard_system = DashboardSystem()
        self.image_analyzer = ImageAnalyzer()
        self.classifier = EnhancedImageClassifier()
        
        # Create demo directories
        self.demo_dir = Path("demo_outputs")
        self.demo_dir.mkdir(exist_ok=True)
        
    def download_sample_images(self):
        """
        Download sample medical images for demonstration
        """
        logger.info("🔄 Downloading sample images for demonstration...")
        
        # Sample medical image URLs (these are placeholder URLs - replace with actual medical image URLs)
        sample_urls = [
            "https://picsum.photos/400/300?random=1",  # Placeholder
            "https://picsum.photos/400/300?random=2",  # Placeholder
            "https://picsum.photos/400/300?random=3",  # Placeholder
        ]
        
        downloaded_images = []
        
        for i, url in enumerate(sample_urls):
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                
                # Save image
                image_path = self.demo_dir / f"sample_image_{i+1}.jpg"
                with open(image_path, 'wb') as f:
                    f.write(response.content)
                
                downloaded_images.append(str(image_path))
                logger.info(f"✅ Downloaded: {image_path}")
                
            except Exception as e:
                logger.error(f"❌ Failed to download {url}: {e}")
        
        return downloaded_images
    
    def run_complete_workflow(self, image_paths):
        """
        Run the complete automated workflow on the images
        """
        logger.info("🚀 Starting complete automated workflow...")
        
        results = []
        
        for i, image_path in enumerate(image_paths):
            logger.info(f"📊 Processing image {i+1}/{len(image_paths)}: {image_path}")
            
            try:
                # Step 1: Enhanced Classification
                logger.info("  🔍 Step 1: Enhanced Classification")
                classification_result = self.classifier.classify_image(image_path)
                
                # Step 2: Image Analysis
                logger.info("  📈 Step 2: Image Analysis")
                analysis_result = self.image_analyzer.analyze_image(image_path)
                
                # Step 3: Dashboard Integration
                logger.info("  📊 Step 3: Dashboard Integration")
                dashboard_result = self.dashboard_system.process_new_image(image_path)
                
                # Combine results
                result = {
                    "image_path": image_path,
                    "classification": classification_result,
                    "analysis": analysis_result,
                    "dashboard": dashboard_result,
                    "timestamp": time.time()
                }
                
                results.append(result)
                logger.info(f"  ✅ Completed processing: {image_path}")
                
            except Exception as e:
                logger.error(f"  ❌ Failed to process {image_path}: {e}")
        
        return results
    
    def generate_demo_report(self, results):
        """
        Generate a comprehensive demo report
        """
        logger.info("📋 Generating demo report...")
        
        report = {
            "demo_info": {
                "total_images_processed": len(results),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "workflow_steps": [
                    "Enhanced Classification",
                    "Image Analysis", 
                    "Visualization Generation",
                    "Dashboard Integration"
                ]
            },
            "results_summary": [],
            "dashboard_report": None
        }
        
        # Process results
        for result in results:
            if "error" not in result["classification"]:
                summary = {
                    "image": Path(result["image_path"]).name,
                    "cancer_type": result["classification"].get("prediction", "unknown"),
                    "confidence": result["classification"].get("confidence", 0.0),
                    "analysis_available": "error" not in result["analysis"],
                    "dashboard_available": "error" not in result["dashboard"]
                }
                report["results_summary"].append(summary)
        
        # Get dashboard report
        try:
            dashboard_report = self.dashboard_system.generate_dashboard_report()
            report["dashboard_report"] = dashboard_report
        except Exception as e:
            logger.error(f"Failed to generate dashboard report: {e}")
        
        return report
    
    def print_demo_summary(self, report):
        """
        Print a formatted demo summary
        """
        print("\n" + "="*60)
        print("🤖 AUTOMATION DEMO SUMMARY")
        print("="*60)
        
        print(f"\n📅 Demo Time: {report['demo_info']['timestamp']}")
        print(f"🖼️  Images Processed: {report['demo_info']['total_images_processed']}")
        
        print(f"\n🔄 Workflow Steps:")
        for i, step in enumerate(report['demo_info']['workflow_steps'], 1):
            print(f"   {i}. {step}")
        
        print(f"\n📊 Results Summary:")
        for result in report['results_summary']:
            confidence_pct = result['confidence'] * 100
            status_icon = "✅" if confidence_pct > 60 else "⚠️"
            print(f"   {status_icon} {result['image']}: {result['cancer_type']} ({confidence_pct:.1f}%)")
        
        if report['dashboard_report'] and 'summary' in report['dashboard_report']:
            summary = report['dashboard_report']['summary']
            print(f"\n📈 Dashboard Statistics:")
            print(f"   Total Images Analyzed: {summary.get('total_images_analyzed', 0)}")
            print(f"   Average Confidence: {summary.get('overall_avg_confidence', 0)*100:.1f}%")
            print(f"   Most Common Type: {summary.get('most_common_cancer_type', 'N/A')}")
            print(f"   System Status: {summary.get('system_status', 'N/A')}")
        
        print(f"\n🎯 Next Steps:")
        print("   1. Visit http://localhost:3000/dashboard to view the dashboard")
        print("   2. Check the 'analysis_outputs' folder for generated visualizations")
        print("   3. Use the API endpoints for further analysis")
        
        print("\n" + "="*60)
    
    def run_demo(self):
        """
        Run the complete automation demo
        """
        logger.info("🎬 Starting Automation Demo...")
        
        try:
            # Step 1: Download sample images
            image_paths = self.download_sample_images()
            
            if not image_paths:
                logger.error("❌ No images downloaded. Demo cannot continue.")
                return
            
            # Step 2: Run complete workflow
            results = self.run_complete_workflow(image_paths)
            
            # Step 3: Generate report
            report = self.generate_demo_report(results)
            
            # Step 4: Print summary
            self.print_demo_summary(report)
            
            logger.info("🎉 Automation Demo completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ Demo failed: {e}")
            raise

def main():
    """
    Main function to run the automation demo
    """
    print("🤖 Cancer Image Scraper - Automation Demo")
    print("=" * 50)
    
    # Check if required directories exist
    required_dirs = ["data", "models", "analysis_outputs"]
    for dir_name in required_dirs:
        Path(dir_name).mkdir(exist_ok=True)
    
    # Run demo
    demo = AutomationDemo()
    demo.run_demo()

if __name__ == "__main__":
    main()
