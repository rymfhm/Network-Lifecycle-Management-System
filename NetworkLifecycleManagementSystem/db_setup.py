from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='db_setup.log',
                    filemode='a')

# Define the base class for SQLAlchemy models
Base = declarative_base()

# Define table models
class NetworkPerformanceLog(Base):
    """Table for storing network performance logs."""
    __tablename__ = 'network_performance_logs'
    id = Column(Integer, primary_key=True, autoincrement=True)
    latency_ms = Column(Float, nullable=False)
    packet_loss = Column(Float, nullable=False)
    bandwidth_mbps = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

class EnergyUsageLog(Base):
    """Table for storing energy usage data."""
    __tablename__ = 'energy_usage_logs'
    id = Column(Integer, primary_key=True, autoincrement=True)
    battery_level = Column(Float, nullable=False)
    energy_consumption = Column(Float, nullable=False)
    renewable_source_percentage = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

class ProcurementLog(Base):
    """Table for storing procurement data."""
    __tablename__ = 'procurement_logs'
    id = Column(Integer, primary_key=True, autoincrement=True)
    item_name = Column(String, nullable=False)
    cost = Column(Float, nullable=False)
    quantity = Column(Integer, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

class BandwidthAllocationLog(Base):
    """Table for storing bandwidth allocation data."""
    __tablename__ = 'bandwidth_allocation_logs'
    id = Column(Integer, primary_key=True, autoincrement=True)
    activity_type = Column(String, nullable=False)
    allocated_bandwidth = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

# Database