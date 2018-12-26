import sys

import mysql.connector
import pandas as pd
from PyQt5 import QtCore, QtWidgets
from mysql.connector import errorcode

from constants import log

Qt = QtCore.Qt
pyqtSignal = QtCore.pyqtSignal


class EDEN_DB(object):
    """
    This is the EDEN database utility
    it is able to connect, create, add, and get from the EDEN MYSQL DATABASE
    After object construction, one must ConnectDatabase, then defineTable before starting working with queries.
    it is still under works [May 2018]
    """

    def __init__(self, username='apaidani_general', password='EDEN17Data',
                 host='distantearths.com', verbose=True, autocommit=False):
        self.db = None
        self.tableName = None
        self.uniques_h = None
        if verbose:
            print("DEFINING CONNECTION")
        self.connection = mysql.connector.connect(user=username, password=password,
                                                  host=host, connect_timeout=5000)
        if verbose:
            print("DEFINING CURSOR")
        self.lastCommand = ''
        self.isClosed = False
        self.verbose = verbose
        self.cursor = self.connection.cursor()
        self.connection.autocommit = autocommit

    def defineTable(self, tableName):
        self.tableName = tableName

    def ConnectDatabase(self, db):
        """
        Connect to given database name in the current account
        :param db: database name
        """
        self.db = db
        try:
            self.connection.database = self.db
            # Set timeouts for 1 hour
            # self.cursor.execute('SET SESSION connect_timeout=3600')
            # self.cursor.execute('SET SESSION wait_timeout=3600')
            # self.cursor.execute('SET SESSION interactive_timeout=3600')
        except mysql.connector.Error as err:
            if err.errno == errorcode.ER_BAD_DB_ERROR:
                self.CreateDatabase()
                self.connection.database = self.db
            else:
                print(err.msg)

    def ShowDatabases(self):
        """
        return a list of available databases under the current account
        :return: list of database names
        """
        dbs = self.RunCommand("SHOW DATABASES;")
        dbs = [tup[0] for tup in dbs]
        return dbs

    def ShowTables(self):
        """
        return a list of available tables in the selected database
        :return: list of table names
        """
        tables = self.RunCommand("SHOW Tables FROM {};".format(self.db))
        tables = [tup[0] for tup in tables]
        return tables

    def CreateDatabase(self):
        try:
            self.RunCommand("CREATE DATABASE %s DEFAULT CHARACTER SET 'utf8';" % self.db)
        except mysql.connector.Error as err:
            print("Failed creating database: {}".format(err.msg))

    def CreateTable(self):
        """
        This table simply exists to create a template table.
        """
        cmd = (" CREATE TABLE IF NOT EXISTS " + self.tableName + " ("
                                                                 " `ID` int(5) NOT NULL AUTO_INCREMENT,"
                                                                 " `date` date NOT NULL,"
                                                                 " `time` time NOT NULL,"
                                                                 " `message` char(50) NOT NULL,"
                                                                 " PRIMARY KEY (`ID`)"
                                                                 ") ENGINE=InnoDB;")
        self.RunCommand(cmd)

    def GetTable(self, cols='*', condition=None):
        """
        Retrieve Table from database. It will retrieve the database in Pandas.DataFrame format.
        params cols: This defines what columns to select from. eg: FIRST-COLUMN,SECOND-COLUMN
        params condition: This defines a condition to select from. eg:  'ID' < 20
        """
        case1 = "SELECT {} FROM {};".format(cols, self.tableName)
        case2 = "SELECT {} FROM {} WHERE {}".format(cols, self.tableName, condition)
        query = case1 if condition is None else case2
        pdTable = pd.read_sql(query, self.connection)
        col_vals = pdTable.columns
        for header in col_vals:
            cmd = "select case when count(distinct `{0}`)=count(`{0}`) then 'True' else 'False' end from {1};".format(
                header, self.tableName)
            if self.RunCommand(cmd)[0][0] == "True":
                self.uniques_h = header
                break
        return pdTable

    def CreateDataFrame(self, mysql_msg, col_vals=None):
        """
        This is an auxiliary function to translate a mysql query result into a pandas Dataframe.
        This method is deprecated because I started using pandas.read_sql function to do this.
        :param mysql_msg: mysql result from the query
        :param col_vals: column/header of table queried, if None, then it will use last headers from cursor.
        :return: table type pandas Dataframe
        """
        if not col_vals:
            col_names = self.cursor.column_names
        else:
            col_names = col_vals
        col_names = [str(col).strip('`').strip('b').strip("'") for col in col_names]
        for i in range(len(col_names)):
            if bytes is type(col_names[i]):
                col_names[i] = str(col_names[i])
        df = pd.DataFrame(mysql_msg, columns=col_names)
        if len(df):
            first_h = list(df)[0]
            if "ID" in first_h:
                try:
                    df[first_h] = df[first_h].astype(int)
                except:
                    pass
        return df

    def GetColumns(self):
        """
        Get table describing the columns of the current selected table.
        :return: pandas Dataframe describing the current table
        """
        return pd.read_sql("SHOW COLUMNS FROM %s;" % self.tableName, self.connection)
        # msg_rows = self.RunCommand("SHOW COLUMNS FROM %s;" % self.tableName)
        # return self.CreateDataFrame(msg_rows)

    def RunCommand(self, cmd):
        """
        Execute raw SQL command. Return result from query
        :param cmd: command
        :return: mysql message
        """
        if self.verbose:
            print("RUNNING COMMAND:\n" + cmd)
        try:
            self.cursor.execute(cmd)
        except mysql.connector.Error as err:
            print('ERROR MESSAGE: ' + str(err.msg))
            print('WITH ' + cmd)
        try:
            msg = self.cursor.fetchall()
        except:
            msg = self.cursor.fetchone()
        self.lastCommand = cmd
        return msg

    def AddEntryToTable(self, columns, values):
        """
        Add an entry to the current database's table. columns and values must be the same size.
        :param columns: column names to which add the `values`
        :param values: values per column
        """
        cmd = "INSERT INTO {} ({})".format(self.tableName, columns)
        cmd += " VALUES ({});".format(values)
        self.RunCommand(cmd)

    def Delete(self, condition):
        """
        Delete rows in current table given a condition to find the values
        :param condition: condition to find the values: e.g. `ID` < 20
        """
        cmd = "DELETE FROM {} WHERE {};".format(self.tableName, condition)
        self.RunCommand(cmd)

    def UpdateVal(self, newvals, conditions):
        """
        Update values in table. Given new values and a condition to find the rows.
        :param newvals: new values format: `Column-Name` = 'new value'
        :param conditions: conditions to find values to be changed: e.g. `ID` < 20
        """
        cmd = "UPDATE {} SET {} WHERE {};".format(self.tableName, newvals, conditions)
        self.RunCommand(cmd)

    def close(self, commit=False):
        """
        Close EDEN MySQL Connection.
        :param commit: if True, all changes will be committed to the MySQL database.
        """
        self.cursor.close()
        if commit:
            self.connection.commit()
        self.connection.close()
        self.isClosed = True

    def __del__(self):
        if not self.isClosed:
            self.close()


class PandaSignal(QtCore.QObject):
    changedData = pyqtSignal(QtCore.QModelIndex, str, str)


class PandasModel(QtCore.QAbstractTableModel, QtCore.QObject):
    """
   Class to populate a PyQt table view with a pandas dataframe
   """

    # changedData = pyqtSignal(QtCore.QModelIndex, str, str)
    # dataChanged = pyqtSignal(QtCore.QModelIndex, str, str)
    # changedData = pyqtSignal('QModelIndex', str, str)
    def __init__(self, data, parent=None):
        QtCore.QAbstractTableModel.__init__(self, parent)
        # self.dataChanged = pyqtSignal(QtCore.QModelIndex, str, str)
        self.signals = PandaSignal()
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parent=None):
        return self._data.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.DisplayRole:
                return str(self._data.iloc[index.row(), index.column()])
            return None

    def headerData(self, col, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self._data.columns[col]
        if role == Qt.ToolTipRole:
            if orientation == Qt.Horizontal:
                return self._data.columns[col]
        return None

    def setData(self, index, value, role=Qt.EditRole):
        if role == Qt.EditRole:
            before = self._data.iloc[index.row(), index.column()]
            after = value
            self._data.iloc[index.row(), index.column()] = value
            # self.signals.dataChanged.emit(index, str(before), str(after))
            self.signals.changedData.emit(index, str(before), str(after))
            log("SIGNAL FOR changedData emmited")
            return True
        return False

    def flags(self, index):
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable


class Completer(QtWidgets.QCompleter):
    def __init__(self, strList, parent=None):
        super(Completer, self).__init__(strList, parent)
        self.setCaseSensitivity(Qt.CaseInsensitive)
        self.setCompletionMode(QtWidgets.QCompleter.PopupCompletion)
        self.setWrapAround(False)

    # Add texts instead of replace
    def pathFromIndex(self, index):
        path = QtWidgets.QCompleter.pathFromIndex(self, index)
        lst = str(self.widget().text()).split(',')
        if len(lst) > 1:
            path = '%s, %s' % (','.join(lst[:-1]), path)
        return path

    # Add operator to separate between texts
    def splitPath(self, path):
        path = str(path.split(',')[-1]).lstrip(' ')
        return [path]


if __name__ == '__main__':
    """
    connection = mysql.connector.connect(user='apaidani_general', password='EDEN17Data',
                              host='distantearths.com')
    cursor = connection.cursor()
    try:
        cursor.execute("SHOW DATABASES;")
    except mysql.connector.Error as err:
        print("ERROR MSG: " + str(err.msg))
    rows = cursor.fetchall()
    rows = [tup[0] for tup in rows]
    print(rows)
    print(type(rows))
    database='apaidani_edendata'
    """
    myEdencnx = EDEN_DB()
    print(myEdencnx.ShowDatabases())
    myEdencnx.ConnectDatabase('apaidani_edendata')
    print(myEdencnx.ShowTables())
    myEdencnx.defineTable('`EDEN Data Files`')
    print(myEdencnx.GetColumns())
    table = myEdencnx.GetTable()
    del myEdencnx
    print(table)
    application = QtWidgets.QApplication(sys.argv)
    view = QtWidgets.QTableView()
    model = PandasModel(table)
    view.setModel(model)
    headers = list(model._data)
    view.show()
    sys.exit(application.exec_())
