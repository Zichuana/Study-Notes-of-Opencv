## Study Notes of Opencv
>杂乱无章的个人笔记  

### 图像拼接
`b.jpg`为主要图像，`a.jpg`为拼接图像。  
- 通过`Image_mosaic.py`进行拼接  
生成`simple-panorma.png`，`best-panorma.png`，后者为最终拼接结果。  
![image](./best-panorma.png)
- 通过`Image_mosaic_new.py`进行拼接  
与上方法几乎一致，但是不会旋转拼接。  
![image](./new.png)
- 通过`Image_mosaic_.py`进行暴力匹配  
生成结果如`violence.png`所示  
![image](./violence.png)  
### 模板匹配
>单目标

- 转灰度匹配
`template_matching_gray`  
结果如图`tematch_gray.png`所示  
![image](./tematch_gray.png)
- RGB匹配
`template_matching_rgb`  
结果如图`tematch_rgb1.png`所示  
![image](./tematch_rgb1.png)
修改通道后如图`tematch_rgb2.png`所示  
![image](./tematch_rgb2.png)
