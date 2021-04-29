---
layout: post
comments: true
title: Cybos Plus 자동매매
categories: Python
tags:
- Python
- cybos
- 자동매매

---

Cybos Plus 자동매매
=======

Cybos Plus에서 사용되는 문법들에 대한 정리본입니다. 


# 연결 여부 체크
objCpCybos = win32com.client.Dispatch("CpUtil.CpCybos")
bConnect = objCpCybos.IsConnect


CpEvent - 실시간 이벤트 수신 (현재가와 주문 체결 실시간 처리)

Cp6033 - 주식 잔고 조회 

CpRPCurrentPrice - 현재가 한 종목 조회

CpMarketEye - 복수 현재가 종목 조회

CpTd6033 - 계좌 잔고 조회

CpTdNew5331A - 계좌 예수금 조회

CpTd5339 - 미체결 조회

ade.CpTd5341 - 주문/체결 내역 조회