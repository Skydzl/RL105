#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 17:23:20 2023

@author: zby
"""
import torch


a=torch.tensor([1,2,3,4,5], requires_grad=True, dtype=torch.float32)
mask=torch.tensor([1,1,0,0,0]) 
b=a.masked_fill(1-mask, -float('inf')) 
# tensor([1., 2., -inf, -inf]) torch.softmax(b, dim=0) 
# tensor([0.2689, 0.7311, 0.0000, 0.0000])
loss = torch.mean(b)
loss.backward()
print(a.grad)