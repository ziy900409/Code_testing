urls = QList()
    urls << QUrl.fromLocalFile("/Users/foo/Code/qt5")
         << QUrl.fromLocalFile(QStandardPaths.standardLocations(QStandardPaths.MusicLocation).first())
    dialog = QFileDialog()
    dialog.setSidebarUrls(urls)
    dialog.setFileMode(QFileDialog.AnyFile)
    if dialog.exec():
        # ...