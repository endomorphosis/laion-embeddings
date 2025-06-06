#!/usr/bin/env python3
"""
Vector Database Assessment Script

This script analyzes the current state of vector database support in the project
and provides specific recommendations for implementation.
"""

import os
import ast
import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class VectorDBAnalysis:
    """Analysis results for a vector database implementation."""
    name: str
    implementation_files: List[str]
    integration_level: str  # "none", "basic", "partial", "complete"
    features_found: List[str]
    missing_features: List[str]
    health_score: int  # 0-100
    recommendations: List[str]


@dataclass
class OverallAssessment:
    """Overall assessment of vector database support."""
    timestamp: str
    databases: Dict[str, VectorDBAnalysis]
    integration_gaps: List[str]
    priority_recommendations: List[str]
    estimated_effort_days: int


class VectorDatabaseAssessment:
    """Assess vector database implementation status."""
    
    def __init__(self, project_root: str = "/home/barberb/laion-embeddings-1"):
        self.project_root = Path(project_root)
        self.analysis_results = {}
        
        # Expected features for each vector database
        self.expected_features = {
            "qdrant": [
                "collection_management", "vector_upload", "vector_search", 
                "hybrid_search", "filtering", "clustering", "health_check",
                "batch_operations", "async_operations", "connection_pooling"
            ],
            "elasticsearch": [
                "index_management", "vector_field_mapping", "knn_search",
                "hybrid_search", "aggregations", "filtering", "bulk_operations",
                "cluster_health", "async_operations", "connection_pooling"
            ],
            "pgvector": [
                "table_management", "vector_column", "similarity_search",
                "hybrid_search_fts", "indexing_hnsw", "filtering_sql",
                "connection_pooling", "async_operations", "migrations"
            ],
            "faiss": [
                "index_creation", "vector_add", "vector_search", 
                "index_persistence", "gpu_support", "quantization",
                "batch_operations", "memory_mapping", "distributed_search"
            ]
        }
    
    def find_implementation_files(self, database: str) -> List[str]:
        """Find files containing implementation for a specific database."""
        search_patterns = {
            "qdrant": [r"qdrant", r"Qdrant"],
            "elasticsearch": [r"elasticsearch", r"Elasticsearch", r"opensearch", r"OpenSearch"],
            "pgvector": [r"pgvector", r"postgres", r"PGVector", r"PostgreSQL"],
            "faiss": [r"faiss", r"FAISS", r"vector_service"]
        }
        
        patterns = search_patterns.get(database, [database])
        found_files = []
        
        # Search Python files
        for py_file in self.project_root.rglob("*.py"):
            try:
                content = py_file.read_text(encoding='utf-8')
                for pattern in patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        found_files.append(str(py_file.relative_to(self.project_root)))
                        break
            except (UnicodeDecodeError, FileNotFoundError):
                continue
        
        return sorted(list(set(found_files)))
    
    def analyze_file_features(self, file_path: Path, database: str) -> List[str]:
        """Analyze a file to find implemented features for a database."""
        try:
            content = file_path.read_text(encoding='utf-8')
        except (UnicodeDecodeError, FileNotFoundError):
            return []
        
        found_features = []
        
        # Database-specific feature detection patterns
        feature_patterns = {
            "qdrant": {
                "collection_management": [r"create_collection", r"delete_collection", r"list_collections"],
                "vector_upload": [r"upload_points", r"add_vectors", r"upsert"],
                "vector_search": [r"search", r"query_points"],
                "hybrid_search": [r"hybrid", r"sparse.*dense"],
                "filtering": [r"filter", r"Filter"],
                "health_check": [r"health", r"ping"],
                "batch_operations": [r"batch", r"bulk"],
                "async_operations": [r"async", r"await"],
                "connection_pooling": [r"pool", r"connection"]
            },
            "elasticsearch": {
                "index_management": [r"create_index", r"delete_index", r"mapping"],
                "vector_field_mapping": [r"dense_vector", r"knn"],
                "knn_search": [r"knn", r"vector.*search"],
                "hybrid_search": [r"hybrid", r"bool.*query"],
                "bulk_operations": [r"bulk", r"batch"],
                "cluster_health": [r"health", r"cluster"],
                "async_operations": [r"async", r"await"],
                "connection_pooling": [r"pool", r"connection"]
            },
            "pgvector": {
                "table_management": [r"CREATE TABLE", r"ALTER TABLE"],
                "vector_column": [r"vector", r"embedding"],
                "similarity_search": [r"<->", r"<#>", r"<=>", r"cosine_distance"],
                "hybrid_search_fts": [r"ts_rank", r"@@", r"to_tsvector"],
                "indexing_hnsw": [r"hnsw", r"ivfflat"],
                "connection_pooling": [r"pool", r"session"],
                "async_operations": [r"async", r"await"],
                "migrations": [r"migration", r"alembic"]
            },
            "faiss": {
                "index_creation": [r"Index", r"IndexFlat", r"IndexIVF"],
                "vector_add": [r"add", r"add_with_ids"],
                "vector_search": [r"search", r"query"],
                "index_persistence": [r"save", r"load", r"write_index"],
                "gpu_support": [r"gpu", r"cuda", r"GpuResources"],
                "quantization": [r"PQ", r"ProductQuantiz"],
                "batch_operations": [r"batch", r"batch_size"],
                "memory_mapping": [r"mmap", r"memory.*map"],
                "distributed_search": [r"distributed", r"shard"]
            }
        }
        
        patterns = feature_patterns.get(database, {})
        
        for feature, regexes in patterns.items():
            for regex in regexes:
                if re.search(regex, content, re.IGNORECASE):
                    found_features.append(feature)
                    break
        
        return found_features
    
    def calculate_health_score(self, found_features: List[str], expected_features: List[str]) -> int:
        """Calculate health score based on implemented features."""
        if not expected_features:
            return 0
        
        implemented_count = len(set(found_features) & set(expected_features))
        total_count = len(expected_features)
        
        base_score = int((implemented_count / total_count) * 100)
        
        # Bonus points for critical features
        critical_features = {
            "qdrant": ["vector_search", "collection_management"],
            "elasticsearch": ["knn_search", "index_management"],
            "pgvector": ["similarity_search", "table_management"],
            "faiss": ["vector_search", "index_creation"]
        }
        
        # Implementation bonus (partial implementation better than none)
        if implemented_count > 0:
            base_score = max(base_score, 20)  # Minimum 20 for any implementation
        
        return min(base_score, 100)
    
    def determine_integration_level(self, health_score: int, file_count: int) -> str:
        """Determine integration level based on health score and file count."""
        if health_score >= 80 and file_count >= 3:
            return "complete"
        elif health_score >= 60 and file_count >= 2:
            return "partial"
        elif health_score >= 20 or file_count >= 1:
            return "basic"
        else:
            return "none"
    
    def generate_recommendations(self, database: str, analysis: VectorDBAnalysis) -> List[str]:
        """Generate specific recommendations for a database."""
        recommendations = []
        
        if analysis.integration_level == "none":
            recommendations.append(f"Implement basic {database} integration from scratch")
            recommendations.append(f"Create {database} configuration and connection management")
            recommendations.append(f"Add {database} to vector store factory")
        
        elif analysis.integration_level == "basic":
            recommendations.append(f"Enhance existing {database} implementation")
            recommendations.append(f"Add missing critical features: {', '.join(analysis.missing_features[:3])}")
            recommendations.append(f"Implement health monitoring for {database}")
        
        elif analysis.integration_level == "partial":
            recommendations.append(f"Complete {database} feature set")
            recommendations.append(f"Add production-ready features: connection pooling, error handling")
            recommendations.append(f"Implement performance optimization for {database}")
        
        else:  # complete
            recommendations.append(f"Maintain and optimize existing {database} implementation")
            recommendations.append(f"Add advanced features and performance monitoring")
            recommendations.append(f"Create comprehensive tests for {database}")
        
        # Add specific missing feature recommendations
        critical_missing = set(analysis.missing_features) & {
            "vector_search", "collection_management", "index_management", 
            "similarity_search", "health_check"
        }
        
        if critical_missing:
            recommendations.append(f"CRITICAL: Implement missing core features: {', '.join(critical_missing)}")
        
        return recommendations
    
    def analyze_database(self, database: str) -> VectorDBAnalysis:
        """Perform complete analysis for a single database."""
        print(f"Analyzing {database} implementation...")
        
        # Find implementation files
        impl_files = self.find_implementation_files(database)
        
        # Analyze features in each file
        all_features = []
        for file_path_str in impl_files:
            file_path = self.project_root / file_path_str
            features = self.analyze_file_features(file_path, database)
            all_features.extend(features)
        
        # Remove duplicates and get unique features
        found_features = list(set(all_features))
        expected_features = self.expected_features.get(database, [])
        missing_features = list(set(expected_features) - set(found_features))
        
        # Calculate health score
        health_score = self.calculate_health_score(found_features, expected_features)
        
        # Determine integration level
        integration_level = self.determine_integration_level(health_score, len(impl_files))
        
        # Create analysis object
        analysis = VectorDBAnalysis(
            name=database,
            implementation_files=impl_files,
            integration_level=integration_level,
            features_found=found_features,
            missing_features=missing_features,
            health_score=health_score,
            recommendations=[]
        )
        
        # Generate recommendations
        analysis.recommendations = self.generate_recommendations(database, analysis)
        
        return analysis
    
    def assess_integration_gaps(self) -> List[str]:
        """Identify overall integration gaps across all databases."""
        gaps = []
        
        # Check for unified interfaces
        unified_files = [
            "services/vector_store_factory.py",
            "services/base_vector_store.py", 
            "services/unified_vector_service.py",
            "services/vector_config.py"
        ]
        
        for file_path in unified_files:
            if not (self.project_root / file_path).exists():
                gaps.append(f"Missing unified interface: {file_path}")
        
        # Check for configuration management
        config_files = [
            "config/vector_databases.yaml",
            "config/vector_databases.json",
            "services/vector_config.py"
        ]
        
        if not any((self.project_root / f).exists() for f in config_files):
            gaps.append("Missing centralized vector database configuration")
        
        # Check for health monitoring
        health_patterns = ["health_monitor", "HealthMonitor", "vector.*health"]
        health_found = False
        
        for py_file in self.project_root.rglob("*.py"):
            try:
                content = py_file.read_text(encoding='utf-8')
                for pattern in health_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        health_found = True
                        break
                if health_found:
                    break
            except (UnicodeDecodeError, FileNotFoundError):
                continue
        
        if not health_found:
            gaps.append("Missing unified health monitoring system")
        
        # Check for migration tools
        migration_files = ["tools/vector_migration.py", "scripts/migrate_vector_db.py"]
        if not any((self.project_root / f).exists() for f in migration_files):
            gaps.append("Missing vector database migration tools")
        
        return gaps
    
    def generate_priority_recommendations(self) -> Tuple[List[str], int]:
        """Generate priority recommendations and effort estimation."""
        recommendations = []
        total_effort_days = 0
        
        # Analyze completion status
        complete_dbs = [db for db, analysis in self.analysis_results.items() 
                       if analysis.integration_level == "complete"]
        partial_dbs = [db for db, analysis in self.analysis_results.items() 
                      if analysis.integration_level == "partial"]
        basic_dbs = [db for db, analysis in self.analysis_results.items() 
                    if analysis.integration_level == "basic"]
        missing_dbs = [db for db, analysis in self.analysis_results.items() 
                      if analysis.integration_level == "none"]
        
        # Priority 1: Critical missing implementations
        if missing_dbs:
            recommendations.append(f"🔴 CRITICAL: Implement missing databases: {', '.join(missing_dbs)}")
            total_effort_days += len(missing_dbs) * 3  # 3 days per new implementation
        
        # Priority 2: Complete partial implementations
        if basic_dbs:
            recommendations.append(f"🟡 HIGH: Complete basic implementations: {', '.join(basic_dbs)}")
            total_effort_days += len(basic_dbs) * 2  # 2 days per basic completion
        
        # Priority 3: Enhance partial implementations
        if partial_dbs:
            recommendations.append(f"🟢 MEDIUM: Enhance partial implementations: {', '.join(partial_dbs)}")
            total_effort_days += len(partial_dbs) * 1  # 1 day per enhancement
        
        # Priority 4: Unified architecture
        recommendations.append("🔴 CRITICAL: Create unified vector store factory and interfaces")
        total_effort_days += 3
        
        recommendations.append("🟡 HIGH: Implement centralized configuration management")
        total_effort_days += 2
        
        recommendations.append("🟡 HIGH: Add comprehensive health monitoring system")
        total_effort_days += 2
        
        # Priority 5: Testing and documentation
        recommendations.append("🟢 MEDIUM: Create comprehensive integration test suite")
        total_effort_days += 3
        
        recommendations.append("🟢 LOW: Complete documentation and migration tools")
        total_effort_days += 2
        
        return recommendations, total_effort_days
    
    def run_assessment(self) -> OverallAssessment:
        """Run complete assessment of all vector databases."""
        print("Starting comprehensive vector database assessment...")
        print("=" * 60)
        
        databases = ["qdrant", "elasticsearch", "pgvector", "faiss"]
        
        # Analyze each database
        for database in databases:
            analysis = self.analyze_database(database)
            self.analysis_results[database] = analysis
            
            print(f"\n{database.upper()}: {analysis.integration_level} "
                  f"(Health: {analysis.health_score}%)")
            print(f"  Files: {len(analysis.implementation_files)}")
            print(f"  Features: {len(analysis.features_found)}/{len(self.expected_features[database])}")
            
        print("\n" + "=" * 60)
        
        # Assess integration gaps
        integration_gaps = self.assess_integration_gaps()
        
        # Generate priority recommendations
        priority_recommendations, effort_days = self.generate_priority_recommendations()
        
        # Create overall assessment
        assessment = OverallAssessment(
            timestamp=datetime.now().isoformat(),
            databases=self.analysis_results,
            integration_gaps=integration_gaps,
            priority_recommendations=priority_recommendations,
            estimated_effort_days=effort_days
        )
        
        return assessment
    
    def save_assessment(self, assessment: OverallAssessment, output_file: str = None):
        """Save assessment results to JSON file."""
        if output_file is None:
            output_file = self.project_root / "docs/vector_database_assessment.json"
        
        # Convert dataclasses to dict for JSON serialization
        assessment_dict = asdict(assessment)
        
        with open(output_file, 'w') as f:
            json.dump(assessment_dict, f, indent=2)
        
        print(f"\nAssessment saved to: {output_file}")
    
    def print_summary(self, assessment: OverallAssessment):
        """Print a human-readable summary of the assessment."""
        print("\n" + "🔍 VECTOR DATABASE ASSESSMENT SUMMARY" + "\n")
        print("=" * 60)
        
        print("\n📊 Database Status:")
        for db_name, analysis in assessment.databases.items():
            status_emoji = {
                "complete": "✅",
                "partial": "🟡", 
                "basic": "🟠",
                "none": "❌"
            }.get(analysis.integration_level, "❓")
            
            print(f"  {status_emoji} {db_name.upper()}: {analysis.integration_level} "
                  f"({analysis.health_score}% health)")
            print(f"     📁 {len(analysis.implementation_files)} files, "
                  f"🔧 {len(analysis.features_found)} features")
        
        print(f"\n🔄 Integration Gaps ({len(assessment.integration_gaps)}):")
        for gap in assessment.integration_gaps:
            print(f"  ❌ {gap}")
        
        print(f"\n⚡ Priority Recommendations:")
        for rec in assessment.priority_recommendations:
            print(f"  {rec}")
        
        print(f"\n⏱️  Estimated Implementation Effort: {assessment.estimated_effort_days} days")
        
        print("\n" + "=" * 60)
        print("💡 Next Steps:")
        print("1. Review detailed recommendations in assessment.json")
        print("2. Start with critical missing implementations")
        print("3. Create unified architecture components")
        print("4. Implement comprehensive testing")


def main():
    """Main function to run the assessment."""
    assessor = VectorDatabaseAssessment()
    
    try:
        # Run the assessment
        assessment = assessor.run_assessment()
        
        # Print summary
        assessor.print_summary(assessment)
        
        # Save detailed results
        assessor.save_assessment(assessment)
        
        print("\n✅ Assessment completed successfully!")
        
    except Exception as e:
        print(f"❌ Assessment failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
