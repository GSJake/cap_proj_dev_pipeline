# Databricks notebook source
# MAGIC %pip install paramiko pandas pyspark

# COMMAND ----------

# COMMAND ----------
# Import required libraries
import dlt
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import paramiko
from io import StringIO
import logging
from datetime import datetime
from databricks.sdk.runtime import *
import zipfile
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# COMMAND ----------

# SFTP Configuration - use Databricks secrets for production
SFTP_CONFIG = {
    'host': 'sftp.greystar.com',
    'port': 22,
    'username': dbutils.secrets.get(scope="gs-sftp", key="username"),
    'password': dbutils.secrets.get(scope="gs-sftp", key="password")
}

# COMMAND ----------

def fetch_and_save_sftp_file(file_path, dbfs_temp_path):
    """
    Connects to SFTP, fetches a ZIP file, extracts the CSV, and saves it to a temporary DBFS path.
    This is a robust, one-time operation per pipeline run.
    """
    ssh = None
    sftp = None

    try:
        # Create SSH client
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        # Connect to SFTP server
        ssh.connect(
            hostname=SFTP_CONFIG['host'],
            port=SFTP_CONFIG['port'],
            username=SFTP_CONFIG['username'],
            password=SFTP_CONFIG['password']
        )
        sftp = ssh.open_sftp()
        logger.info(f"Connected to SFTP server: {SFTP_CONFIG['host']}")

        # Read the remote ZIP file's content as binary
        with sftp.file(file_path, 'rb') as remote_file:
            zip_content = remote_file.read()

        # Extract CSV from ZIP file
        with zipfile.ZipFile(BytesIO(zip_content)) as zip_file:
            # Get list of files in the ZIP
            file_list = zip_file.namelist()
            logger.info(f"Files in ZIP: {file_list}")
            
            # Find the CSV file (assuming there's only one CSV or we want the first one)
            csv_file = None
            for filename in file_list:
                if filename.lower().endswith('.csv'):
                    csv_file = filename
                    break
            
            if csv_file is None:
                raise ValueError("No CSV file found in the ZIP archive")
            
            # Extract and read the CSV content
            csv_content = zip_file.read(csv_file).decode('utf-8')

        # Write the CSV content to a temporary location on DBFS
        spark = SparkSession.getActiveSession()
        dbutils.fs.put(dbfs_temp_path, csv_content, overwrite=True)
        logger.info(f"Successfully extracted CSV from ZIP and saved to DBFS at {dbfs_temp_path}")

    except Exception as e:
        logger.error(f"Failed to fetch data from SFTP: {str(e)}")
        raise
    finally:
        if sftp:
            sftp.close()
        if ssh:
            ssh.close()

# COMMAND ----------

# Bronze Table - Raw data ingestion
@dlt.table(
    name="raw_cap_projects",
    comment="Raw capital projects data ingested from SFTP server",
    table_properties={
        "quality": "bronze",
        "pipelines.autoOptimize.managed": "true"
    }
)
def raw_cap_projects():
    """
    Bronze table: Raw data from SFTP with minimal processing.
    This function reads the file from DBFS after it's been downloaded.
    """
    # The actual file path on the SFTP server - dynamically generated with current date
    current_date = datetime.now().strftime('%Y%m%d')
    file_path = f"CapitalProjects__{current_date}.csv.zip"

    # Create a unique temporary path on DBFS to store the fetched file
    dbfs_temp_path = f"/FileStore/sftp_temp/{file_path.replace('/', '_')}-{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"

    # Fetch the file from SFTP and save it to DBFS
    fetch_and_save_sftp_file(file_path, dbfs_temp_path)

    # Use Spark to read the first few lines to determine the header
    lines_df = spark.read.text(dbfs_temp_path).limit(4)
    # The actual column headers are on the second line of the file
    header_line = lines_df.collect()[2]['value'] 
    
    # Create a list of cleaned column names
    column_names = [
        c.strip().strip('"').replace(' ', '_').replace('.', '').replace('#', '')
        for c in header_line.split(',')
    ]

    # Read the actual CSV data, skipping the first two rows (which are headers)
    # The data will have generic column names like _c0, _c1, etc.
    df = (
        spark.read.format("csv")
        .option("header", "false") # We handle the header manually
        .option("skipRows", 3)
        .option("inferSchema", "true")
        .option("delimiter", ",")
        .load(dbfs_temp_path)
    )

    # Apply the cleaned column names to the DataFrame using toDF()
    final_df = df.toDF(*column_names)

    # Clean up the temporary file on DBFS after reading is complete

    logger.info(f"Cleaned up temporary file at {dbfs_temp_path}")

    # Add metadata columns and return the final raw DataFrame
    return final_df.withColumn("ingestion_timestamp", current_timestamp()) \
                   .withColumn("source_file", lit(file_path)) \
                   .withColumn("pipeline_id", lit("cap_projects_dlt_pipeline")) \
                   .withColumn("ingestion_date", to_date(current_timestamp()))

# COMMAND ----------

@dlt.table(
    name="silver_cap_projects",
    comment="Silver table: cleaned and typed Capital Projects data",
    table_properties={"quality": "silver"}
)
def clean_cap_projects():
    """
    Silver table: Cleans and types the data from the bronze table.
    This function now works with clean column names and handles date parsing.
    """
    df = dlt.read("raw_cap_projects")

    return df.select(
        col("IRN").cast("long").alias('irn'),
        
        # Parse datetime string and convert to date
        to_date(col("Invoice_Date"), "MM/dd/yyyy HH:mm:ss").alias("invoice_date"),
        
        col("Vendor_Code").alias("vendor_code"),
        col("Supplier_Name").alias("supplier_name"),
        col("Property_Name").alias("property_name"),
        col("Invoice_Number").alias("invoice_number"),
        regexp_replace(col("Grand_Total"), "[$,]", "").cast("double").alias("grand_total"),
        col("Invoice_Status").alias("invoice_status"),
        col("Check_Number").alias("check_number"),
        
        # Parse datetime string and convert to date
        to_date(col("Check_Date"), "MM/dd/yyyy HH:mm:ss").alias("check_date"),
        
        col("Payment_Status").alias("payment_status"),
        "ingestion_date",
        "source_file"
    )