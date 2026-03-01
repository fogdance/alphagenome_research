"""
Database connector for continuous contract mapping.
"""

import os
from dataclasses import dataclass
from typing import List, Tuple
import pymysql
import pandas as pd


@dataclass
class DBConfig:
    host: str
    port: int
    db: str
    user: str
    password: str


# Default config from environment
DB_CONFIG = DBConfig(
    host=os.getenv("MD_MYSQL_HOST", "10.0.0.9"),
    port=int(os.getenv("MD_MYSQL_PORT", "3306")),
    db=os.getenv("MD_MYSQL_DB", "market_data"),
    user=os.getenv("MD_MYSQL_USER", "root"),
    password=os.getenv("MD_MYSQL_PASSWORD", "root"),
)


def get_connection(config: DBConfig = None):
    """Get MySQL connection."""
    if config is None:
        config = DB_CONFIG
    
    return pymysql.connect(
        host=config.host,
        port=config.port,
        user=config.user,
        password=config.password,
        database=config.db,
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )


def get_continuous_map(csymbol: str, start_date: str = None, end_date: str = None, config: DBConfig = None) -> pd.DataFrame:
    """
    Get continuous contract mapping from database.
    
    Args:
        csymbol: Continuous symbol (e.g., 'DCE.JM')
        start_date: Start date (YYYY-MM-DD), optional
        end_date: End date (YYYY-MM-DD), optional
        config: Database config, optional
    
    Returns:
        DataFrame with columns: csymbol, trading_date, symbol
    """
    conn = get_connection(config)
    
    try:
        query = """
            SELECT csymbol, trading_date, symbol
            FROM fut_continuous_map_v2
            WHERE csymbol = %s
        """
        params = [csymbol]
        
        if start_date:
            query += " AND trading_date >= %s"
            params.append(start_date)
        
        if end_date:
            query += " AND trading_date <= %s"
            params.append(end_date)
        
        query += " ORDER BY trading_date"

        with conn.cursor() as cursor:
            cursor.execute(query, params)
            results = cursor.fetchall()

        df = pd.DataFrame(results)
        if len(df) > 0:
            df['trading_date'] = pd.to_datetime(df['trading_date'])
        
        return df
    
    finally:
        conn.close()


def get_contract_info(symbol: str, config: DBConfig = None) -> dict:
    """
    Get contract information.
    
    Args:
        symbol: Real contract symbol (e.g., 'DCE.jm2505')
        config: Database config, optional
    
    Returns:
        Contract info dict
    """
    conn = get_connection(config)
    
    try:
        query = """
            SELECT *
            FROM fut_contract_v2
            WHERE symbol = %s
        """
        
        with conn.cursor() as cursor:
            cursor.execute(query, [symbol])
            result = cursor.fetchone()
        
        return result
    
    finally:
        conn.close()


def get_all_continuous_symbols(config: DBConfig = None) -> List[str]:
    """
    Get all continuous symbols from database.
    
    Returns:
        List of continuous symbols
    """
    conn = get_connection(config)
    
    try:
        query = """
            SELECT DISTINCT csymbol
            FROM fut_continuous_map_v2
            ORDER BY csymbol
        """
        
        df = pd.read_sql(query, conn)
        return df['csymbol'].tolist()
    
    finally:
        conn.close()
