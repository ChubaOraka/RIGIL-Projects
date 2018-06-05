# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 12:39:15 2018
Test MySQL

@author: Rigilite
"""

from sqlalchemy import create_engine

engine = create_engine("mysql://root@localhost-MySQL5.7.21-['(none)']")
connection = engine.connect()
connection.close()