################################################################################
#
#
#
#       acquire.py
#
#       Description: Description
#
#       Class:
#
#           AcquireZillow
#
#       Class Fields:
#
#           file_name
#           database_name
#           sql
#
#       Class Methods:
#
#           __init__(self)
#           pre_preparation(self, df)
#
#       Inherited Methods:
#
#           get_data(self, use_cache = True, cache_data = True)
#
#
################################################################################

import pandas as pd

try:
    from _acquire import Acquire
except ModuleNotFoundError:
    from util.acquire import Acquire

################################################################################

class AcquireZillow(Acquire):

    ################################################################################

    def __init__(self):

        
        self.file_name = 'zillow.csv'
        self.database_name = 'zillow'
        self.sql = '''
            SELECT
                properties_2017.*,
                logerror,
                transactiondate,
                typeconstructiondesc,
                airconditioningdesc,
                architecturalstyledesc,
                buildingclassdesc,
                propertylandusedesc,
                storydesc,
                heatingorsystemdesc
            FROM properties_2017
            JOIN predictions_2017 ON properties_2017.parcelid = predictions_2017.parcelid
                AND predictions_2017.transactiondate LIKE '2017%%'
            LEFT JOIN typeconstructiontype USING (typeconstructiontypeid)
            LEFT JOIN airconditioningtype USING (airconditioningtypeid)
            LEFT JOIN architecturalstyletype USING (architecturalstyletypeid)
            LEFT JOIN buildingclasstype USING (buildingclasstypeid)
            LEFT JOIN propertylandusetype USING (propertylandusetypeid)
            LEFT JOIN storytype USING (storytypeid)
            LEFT JOIN heatingorsystemtype USING (heatingorsystemtypeid)
                
            JOIN (
                SELECT
                    parcelid,
                    MAX(transactiondate) AS date
                FROM predictions_2017
                GROUP BY parcelid
            ) AS max_dates ON properties_2017.parcelid = max_dates.parcelid
                AND predictions_2017.transactiondate = max_dates.date
                
            WHERE latitude IS NOT NULL AND longitude IS NOT NULL;
        '''

    ################################################################################

    def pre_preparation(self, df: pd.DataFrame) -> pd.DataFrame:
        drop_columns = [
            'heatingorsystemtypeid',
            'storytypeid',
            'propertylandusetypeid',
            'buildingclasstypeid',
            'architecturalstyletypeid',
            'airconditioningtypeid',
            'typeconstructiontypeid',
            'id',
            'parcelid'
        ]

        df = df.drop(columns = drop_columns)
        return df