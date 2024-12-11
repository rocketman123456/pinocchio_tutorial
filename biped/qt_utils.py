from PySide6 import QtCore, QtWidgets


def get_qt_app():
    instance = QtCore.QCoreApplication.instance()
    if instance is None:
        return QtWidgets.QApplication([""])
    else:
        return instance
