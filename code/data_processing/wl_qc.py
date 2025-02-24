import pandas as pd
from data_processing.utils import WL_UTILS, DWL_UTILS

class WL_QC:

    def __init__(self, WL=True):

        self.WL = WL
        self.CATEGORY = 1

    def wl_qc(self, submission, version):
        """

        calls WL_UTILS to get df_all

        parses df_all to categorize

        """
        df = submission

        wl_instance = WL_UTILS()
        df_all, self.CATEGORY = wl_instance.main(df, version)

        if self.CATEGORY == 3:
            print("One or more conditions are empty, status finalized at 3")
            return self.CATEGORY
        # Assuming df_all is the DataFrame and self.CATEGORY exists in the class context

        if (df_all['block'] == 'immediate').any():
            if df_all.loc[df_all['block'] == 'immediate', 'ratio'].iloc[0] < 0.3:
                self.CATEGORY = 2


        return df_all, self.CATEGORY

    def dwl_qc(self, submission, version):
        """

        Calls DWL_UTILS to return master df

        parses df to categorize

        """
        df = submission

        dwl_instance = DWL_UTILS()
        df_all, self.CATEGORY = dwl_instance.main(df, version)

        if self.CATEGORY == 3:
            print("One or more conditions are empty, status finalized at 3")
            return self.CATEGORY
        # Assuming df_all is the DataFrame and self.CATEGORY exists in the class context

        if (df_all['block'] == 'delay').any():
            if df_all.loc[df_all['block'] == 'delay', 'ratio'].iloc[0] < 0.3:
                self.CATEGORY = 2

        return df_all, self.CATEGORY
