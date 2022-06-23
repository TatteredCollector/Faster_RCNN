- 1.Conv layers。    
  作为一种CNN网络目标检测方法，  
  Faster RCNN首先使用一组基础的conv+relu+pooling层提取image的feature maps。   
  该feature maps被共享用于后续RPN层和全连接层。

- 2.Region Proposal Networks。   
  RPN网络用于生成region proposals。    
  该层通过softmax判断anchors属于positive或者negative， 
  再利用bounding box regression修正anchors获得精确的proposals。

- 3.Roi Pooling。    
  该层收集输入的feature maps和proposals， 
  综合这些信息后提取proposal feature maps，送入后续全连接层判定目标类别。

- 4.Classification。 
  利用proposal feature maps计算proposal的类别， 
  同时再次bounding box regression获得检测框最终的精确位置。

- 以VGG16版本为例    
  卷积提取之前，将图像缩放成一致大小：3,M,N 一般M=800，N=600 
  Conver layers 中的卷积层不改变特征图尺寸，池化层 k=2 stride =2 缩减为原特征图的0.5 
  输入进RPN网络的特征图为：512,M/16,N/16

- RPN:  
  anchors的坐标是矩形框的左上角和右下角 [x1,y1,x2,y2]  
  三种尺度与三种比例 尺度：128*128 256*256 512*512(根据归一化的尺度设定？？？)   
  比例： 1:1 1:2 2：1   
  第一步：3*3卷积 输出256,M/16,N/16 
  判断正负类别:   
  1.利用1*1卷积 输出2*9,M/16,N/16     
  2.reshape：2,9*(M/16)*(N/16)   
  3.softmax 区分背景;   
  4.reshape: 1  
  bounding box regression:  
  利用1*1卷积 输出4*9,M/16,N/16

- proposal layer    
  Proposal Layer负责综合所有坐标变换量和positive anchors，   
  计算出精准的proposal，送入后续RoI Pooling Layer。

- with torch.no_grad的作用 
  在该模块下，所有计算得出的tensor的requires_grad都自动设置为False

- 测试运行时间：   
  模型第一次启动时间很慢，  
  算时间应该从第二次正向传播开始

- voc2012数据集网址  
  http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar    
  http://pjreddie.com/projects/pascal-voc-dataset-mirror/