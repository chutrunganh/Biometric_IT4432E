# 2D Face recognition

Lưu ý check lại thông tin, bìa viết đã không được cập nhật trong 8 năm: https://viblo.asia/p/opencv-nhan-dang-face-recognition-ZjlearmzkqJ


# Intro

Face recognition là một lĩnh vực nghiên cứu của ngành Computer Vision, và cũng được xem là một lĩnh vực nghiên cứu của ngành Biometrics (về bản chất gần như tương đồng với Fingerprint hay iris recognition chỉ khác về các feature được extract ra với mỗi loại) 


Finger print thi can extract ra nhưng cái gì, iris thi can extract ra nhưng cái gì, face recognition thi can extract ra nhưng cái gì ????


Face Recognition là bài toán nhận dạng và xác thực người dựa vào khuôn mặt của họ. Đối với con người thì đó là một nhiệm vụ rất đơn giản, thậm chí là ở trong những điều kiện môi trường khác nhau, tuổi tác thay đổi, đội mũ, đeo kính, … Tuy nhiên, đối với máy tính thì nó vẫn còn là một thử thách khó khăn trong vài thập kỷ qua cho đến tận ngày nay. Trong thời đại bùng nổ của trí tuệ nhân tạo, tận dụng sức mạnh của các thuật toán DL và lượng dữ liệu vô cùng lớn, chúng ta có thể tạo ra các models hiện đại, cho phép biểu diễn khuôn mặt thành các vectors đặc trưng trong không gian nhiều chiều. Để từ đó, máy tính có thể thực hiện nhận diện ra từng người riêng biệt, mà thậm chí còn vượt qua khả năng của con người trong một số trường hợp.


Vi du: https://www.forbes.com/sites/amitchowdhry/2014/03/18/facebooks-deepface-software-can-match-faces-with-97-25-accuracy/


Earlier this year Facebook created DeepFace, a facial recognition system almost as accurate as the human brain. In a standardized test involving pairs of photographs, a human being would get the answer correct 97.53% of the time; Facebook’s technology scored an impressive 97.25%. Most people thought that was as far as facial recognition breakthroughs would go in 2014. They were wrong.

Tìm thêm số liệu mới hơn ????


## Thực trạng, Tiền năng của face recognition:

Trong khi các phương pháp sinh trắc học khác như khi nhận dạng vân tay và mống mắt đã đạt tới độ chín, tức là có thể áp dụng trên thực tế một cách rộng rãi thì nhận dạng mặt người vẫn còn nhiều thách thức và vẫn là một lĩnh vực nghiên cứu thú vị với nhiều người.


Không chỉ có tiềm năng trong việc bảo mật mà ta bản tới trong dự án này mà face recognition còn có thể ứng dụng trong nhiều lĩnh vực khác như: giám sát trên quy mô đô thị, giám sát giao thông, giám sát an ninh dặc biệt khi kết hợp với phân tichs dữ liệu lớn và AI . Vd Skynet (Lưới trời) của Trung Quốc ... theo https://trithucvn2.net/trung-quoc/phia-sau-du-an-skynet-cua-trung-quoc-voi-hon-600-trieu-camera-giam-sat.html



Các thiết bị ứng dụng của face recognition: Mớ khóa khuôn mặt 2D trên các thiết bị động android, Windows Hello trên Windows 10, Face ID trên các sản phẩm của Apple, máy chấm công bằng nhận dạng khuôn mặt, camera an ninh, điểm danh,....


Nhận diện khuôn mặt (Face Recognition) là một trong những thách thức lớn mà các nhà nghiên cứu về Học máy - Học sâu đã và đang phải đối mặt. Bài toán này có thể được áp dụng ở rất nhiều lĩnh vực khác nhau, đặc biệt trong những lĩnh vực yêu cầu độ chính xác và bảo mật cao như eKYC trong E-Comercial và nhận diện danh tính qua surveillance camera (CCTV)


## Định hướng phát triển của dự án:

Hiện nay các phương pháp nhận dạng mặt được chia thành nhiều hướng theo các tiêu chí khác nhau: nhận dạng với dữ liệu đầu vào là ảnh tĩnh 2D(still image based FR) là phổ biến nhất, tuy nhiên tương lai có lẽ sẽ là 3D FR (vì việc bố trí nhiều camera 2D sẽ cho dữ liệu 3D và đem lại kết quả

- Áp dụng để tổng hợp ảnh 3D, xử lý trên video thời gian thực, liên kết với dữ liệu dân cư


## Thách thức


Challenges in Recognition Systems

Facing challenges while working on recognition systems in common, all you need to learn is how to get out of them. Here are some common challenges:

    Pose: Recognition systems are susceptible to the human pose. Facial recognition systems will not be able to predict if the person’s face is not visible.
    Illumination: Illumination changes the face contours drastically. Face recognition pictures should be clear with proper brightness.
    Facial Expressions: Different facial expressions can result in different predictions of the same person’s Image.
    Low Resolution: Low-resolution pictures contain less information, hence not good for face recognition training.



## Các hướng nghiên cứu đã có:


Traditional Face Recognition Algorithm

Traditional face recognition algorithms don’t meet modern-day’s facial recognition standards. They were designed to recognize faces using old conventional algorithms.

OpenCV provides some traditional facial Recognition Algorithms.

    Eigenfaces
    Scale Invariant Feature Transform (SIFT)
    Fisher faces
    Local Binary Patterns Histograms (LBPH)

These methods differ in the way they extract image information and match input and output images.

LBPH algorithm is a simple yet very efficient method still in use but it’s slow compared to modern days algorithms.
LBPH algorithm |Face Recognition Algorithm:



Deep Learning For Face Recognition

There are various deep learning-based facial recognition algorithms available.

    DeepFace
    DeepID series of systems,
    FaceNet
    VGGFace

Generally, face recognizers that are based on landmarks take face images and try to find essential feature points such as eyebrows, corners of the mouth, eyes, nose, lips, etc. There are more than 60 points.

https://www.analyticsvidhya.com/blog/2022/04/face-recognition-system-using-python/



Gồm các hướng tiếp cận:

- phuong pháp toàn cục, vd như PCA, LDA, SVM, Neural Network, CNN, RNN, GAN, ...
- phương pháp cục bộ, vd như LBP, HOG, SIFT, SURF, ...
- phương pháp kết hợp, vd như phương pháp kết hợp giữa CNN và RNN, CNN và GAN, ...

Nhưng trong đó, phương pháp sử dụng Deep Learning, đặc biệt là Convolutional Neural Network (CNN) đang là xu hướng phổ biến và hiệu quả nhất hiện nay ????

Các phuowgn pháp cục bộ đã chứng tỏ sư ưu việt  hơn khi làm việc trong các điều kiện không có kiểm soát và có thể nói rằng lịch sử phát triển của nhận dạng mặt (A never ending story) là sự phát triển của các phương pháp trích chọn đặc trưng (feature extractrion methods) được sử dụng trong các hệ thống dựa trên feature based

Ko chắc đúng, check lại ???


Hai bài toán chính trong face recognition:

- Face verification: xác nhận xem hai ảnh có chứa cùng một người hay không (1:1)

- Face recognition: xác định xem ảnh đầu vào chứa người nào trong tập dữ liệu đã biết (1:N)

Còn 1 thuật ngữ nữa là Face detection: là bài toán xác định xem có khuôn mặt trong ảnh hay không, nếu có thì vị trí của nó là gì ???  -> đây chỉ là một bước tiền xử lý cho bài toán nhận dạng mặt


Face Recognition có thể chia thành 3 bài toán nhỏ:

    Face Authentication: Hạn chế quyền truy cập của một người đến một nguồn tài nguyên nào đó.
    Face Verification: Xác nhận một người phù hợp với ID của họ.
    Face Identification: Gán chính xác tên của người.

Ba bài toán này thực ra chỉ khác nhau ở mục đích sử dụng kết quả nhận diện khuôn mặt vào việc gì, còn về bản chất vẫn là phân loại xem khuôn mặt cần nhận diện có thuộc vào nhóm nào trong bộ dữ liệu cho trước hay không?

Tất cả những bài toán này đều cần phải được giải quyết trong cả 3 trường hợp:

    Người trong ảnh
    Người trong file video
    Người thực (stream real-time từ camera)

Tuy nhiên, cũng lại xuất hiện thêm một bài toán con con nữa, đó là đôi khi chúng ta cần phân biệt đâu là người thật, đâu là người giả (người trong video hay ảnh). Vì nếu chúng ta đối xử với cả 3 trường hợp đều như nhau thì rất có thể kẻ gian sẽ lợi dụng để truy cập trái phép vào hệ thống thông qua một bức ảnh, cái mà rất dễ dàng có được.




# IR cam


Camera hồng ngoại (IR) được sử dụng trong mở khóa khuôn mặt vì nó giúp cải thiện tính chính xác và độ an toàn của hệ thống. Dưới đây là một số lý do chính tại sao cần sử dụng camera IR thay vì camera thông thường:

    Hoạt động tốt trong điều kiện ánh sáng yếu hoặc ban đêm:
        Camera thông thường phụ thuộc vào ánh sáng nhìn thấy (visible light), vì vậy khi ánh sáng yếu hoặc tối hoàn toàn, nó sẽ khó có thể nhận diện khuôn mặt chính xác.
        Camera IR phát ra tia hồng ngoại, cho phép nhận diện khuôn mặt trong điều kiện ánh sáng yếu hoặc không có ánh sáng mà không cần đèn nền.

    Phát hiện giả mạo (liveness detection):
        Camera IR có thể phân biệt giữa hình ảnh hoặc video của khuôn mặt và khuôn mặt thật. Nó có thể phát hiện các dấu hiệu như nhiệt độ cơ thể hoặc các chi tiết bề mặt da mà các camera thông thường khó làm được.
        Điều này giúp tránh các trường hợp bị lừa bởi việc sử dụng ảnh chụp hoặc video để đánh lừa hệ thống.

    Giảm thiểu ảnh hưởng của điều kiện ánh sáng thay đổi:
        Camera thông thường dễ bị ảnh hưởng bởi ánh sáng chói hoặc bóng tối, dẫn đến sự không ổn định trong việc nhận diện khuôn mặt.
        Camera IR không bị ảnh hưởng bởi những thay đổi trong ánh sáng xung quanh, giúp duy trì độ chính xác ổn định.

    Tăng cường khả năng bảo mật:
        Sử dụng camera IR cho phép hệ thống phân tích thêm các đặc điểm bề mặt khuôn mặt và kết cấu 3D, làm cho việc giả mạo trở nên khó khăn hơn.
        Một số hệ thống còn sử dụng camera IR để đo độ sâu (depth sensing), giúp tạo ra mô hình 3D của khuôn mặt, làm tăng độ an toàn và chính xác.

Như vậy, camera IR mang lại nhiều lợi ích quan trọng cho hệ thống nhận diện khuôn mặt, đặc biệt là về tính chính xác và bảo mật trong mọi điều kiện ánh sáng.


## Step


Steps Involved in Face Recognition

    Face Detection: Locate the face, note the coordinates of each face locate,d and draw a bounding box around every faces.
    Face Alignments. Normalize the faces in order to attain fast training.
    Feature Extraction. Local feature extraction from facial pictures for training, this step is performed differently by different algorithms.
    Face Recognition. Match the input face with one or more known faces in our dataset.




để xây dựng một hệ thống nhận dạng mặt, ta cần thực hiện các bước sau:

- Step 1: Data collection: thu thập dữ liệu ảnh khuôn mặt từ nhiều nguồn khác nhau
- Step 2: Face detection: xác định vị trí của khuôn mặt trong ảnh và cắt nó ra để xử lý
- Step 3: Data preprocessing: tiền xử lý dữ liệu ảnh khuôn mặt. bao gồm các bước căn chỉnh ảnh (face image alignment) và chuẩn hóa ánh sáng (illumination normalization) (ở đây tôi đang nói tới các ảnh có góc nhìn thẳng – frontal view face image)
- Step 4: Feature extraction: trích chọn đặc trưng từ ảnh khuôn mặt. ở bước này một phương pháp trích chọn đặc điểm nào đó (mẫu nhị phân cục bộ – Local Binary Pattern – LBP, Gabor wavelets, …) sẽ được sử dụng với ảnh mặt để trích xuất các thông tin đặc trưng cho ảnh, kết quả là mỗi ảnh sẽ được biểu diễn dưới dạng một vector đặc điểm (feature vector)

- Step 5: bước nhận dạng (recognition) hay phân lớp (classification), tức là xác định danh tính (identity) hay nhãn (label) của ảnh – đó là ảnh của ai. Các phương pháp vd như: KNN, SVM, Neural Network, CNN, RNN, GAN, ...  thêm vài cái trong khóa Biometric vào đây ?????



Luồng xử lý của bài toán Face Recognition

Bài toán Face Recognition bắt buộc phải bao gồm tối thiếu 3 bước sau:

    Bước 1: Face Detection - Xác định vị trí của khuôn mặt trong ảnh (hoặc video frame). Vùng này sẽ được đánh dấu bằng một hình chữ nhật bao quanh.
    Bước 2: Face Extraction (Face Embedding) - Trích xuất đặc trưng của khuôn mặt thành một vector đặc trưng trong không gian nhiều chiểu (thường là 128 chiều).
    Bước 3: Face Classification (Face Authentication - Face Verification - Face Identification).

Ngoài 3 bước trên, trong thực tế chúng ta thường bổ sung thêm một số bước để tăng độ chính xác nhận diện:

    Image Preprocessing: Xử lý giảm nhiễu, giảm mờ, giảm kích thước, chuyển sang ảnh xám, chuẩn hóa, …
    Face Aligment: Nếu ảnh khuôn mặt bị nghiêng thì căn chỉnh lại sao cho ngay ngắn.
    Kết hợp nhiều phương pháp khác nhau tại bước 3.


![image1](images/image1.png)


Vẽ lại sau ???



### Face Detection

Face Detection là bước đầu tiên trong bài toán Face Recognition, có vai trò rất lớn trong việc nâng cao độ chính xác của toàn bộ hệ thống. Đầu vào của nó là một bức ảnh có chứa mặt người, đầu ra của nó sẽ là các tọa độ của vùng chứa khuôn mặt, thường thể hiện bằng một hình chữ nhật bao quanh khuôn mặt đó.

Có 2 phương pháp tiếp cận để giải quyết vấn đề ở bước này:

    Feature-based: Sử dụng các bộ lọc thủ công (hand-crafted filters) để tìm kiếm và định vị vị trí khuôn mặt trong ảnh. Phương pháp này rất nhanh và hiệu quả trong điều kiện gần lý tưởng, nhưng không hiệu quả trong điều kiện phức tạp hơn.

        Điều kiện gần lý tưởng: Ảnh chất lượng cao, khuôn mặt nằm ở trung tâm, không bị che khuất, không bị nghiêng, không bị mờ, không bị nhiễu, không bị thay đổi ánh sáng, … chỉ có ít khuon mặt trong ảnh.

        Điều kiện phức tạp: Ảnh chất lượng thấp, khuôn mặt nằm ở ngoài lề, bị che khuất, bị nghiêng, bị mờ, bị nhiễu, bị thay đổi ánh sáng, … có nhiều khuôn mặt trong ảnh.
    


    Image-based: Sử dụng các thuật toán DL để học và tự động định vị vị trí khuôn mặt dựa trên toàn bộ bức ảnh. Ưu điểm của phương pháp này là độ chính xác cao hơn so với phương pháp Feature-based, nhưng tốc độ thực hiện thì lại chậm hơn. Tùy theo điều kiện cụ thể của từng bài toán mà ta chọn phương pháp phù hợp. VD: chạy trên thiết bị nào (PC hay Embedded Device), có cần Real-time hay không, điều kiện môi trường xung quanh ra sao, …

Dưới đây là bảng tổng hợp các thư viện và thuật toán cho mỗi phương pháp này:

![face_detector](images/face_detector.png)

Chạy thử trên máy xem thời gian khác bọt như nàO ??? ghi rõ cấu hình, thời gian, ... ???

Nhìn chung, phương pháp Image-based có sử dụng các thuật toán DL nên độ chính xác cao hơn so với phương pháp Feature-based. Nhưng đổi lại, xét về tốc độ thực hiện thì Feature-based lại là kẻ chiến thắng. Tuy nhiên, điều này chỉ biểu hiện rõ rệt nếu chúng ta chạy trên các thiết bị có cấu hình thấp, kiểu như các thiết bị nhúng, còn nếu chạy trên PC hay server thì sự khác biệt về tốc độ thực thi giữa 2 phương pháp là không đáng kể.




### Face Embedding 


Face Embedding

Đây là bước thứ 2 trong bài toán Face Recognition. Input của nó là bức ảnh khuôn mặt đã tìm ra ở bước 1, còn Output là một Vector nhiều chiều thể hiện đặc trưng của khuôn mặt đó.

Hai thuật toán phổ biến nhất hiện nay để thực hiện Face Embedding là FaceNet và VGGNet.

    FaceNet được tạo ra bởi Florian Schroff và đồng nghiệp tại Google. Họ đã miêu tả nó trong bài báo năm 2015 với tiêu đề FaceNet: A Unified Embedding for Face Recognition and Clustering. Ý tưởng của FaceNet được gọi là Triplet Loss, cho phép hình ảnh được mã hóa hiệu quả dưới dạng vectơ đặc trưng, để từ đó tính toán và đối sánh độ tương đồng nhanh chóng thông qua các phép tính khoảng cách trong không gian. Hệ thống của họ đã đạt được kết quả state-of-the-art.

FaceNet, that directly learns a mapping from face images to a compact Euclidean space where distances directly correspond to a measure of face similarity. […] Our method uses a deep convolutional network trained to directly optimize the embedding itself, rather than an intermediate bottleneck layer as in previous deep learning approaches. To train, we use triplets of roughly aligned matching / non-matching face patches generated using a novel online triplet mining method. — FaceNet: A Unified Embedding for Face Recognition and Clustering, 2015.

    VGGFace được phát triển bởi Omkar Parkhi và đồng nghiệp từ Visual Geometry Group (VGG) tại Oxford. Nó được mô tả trong bài báo năm 2015 của họ có tiêu đề Deep Face Recognition. Trọng tâm chính của họ là thu thập một tập dữ liệu đào tạo rất lớn và sử dụng tập dữ liệu này để đào tạo một mô hình CNN rất sâu về khả năng nhận diện khuôn mặt.

… we show how a very large scale dataset (2.6M images, over 2.6K people) can be assembled by a combination of automation and human in the loop — Deep Face Recognition, 2015.

Cả 2 thuật toán này đều có Pre-trained model. Chúng ta hoàn toàn có thể sử dụng chúng một cách miễn phí trong các dự án của mình. Mình sẽ đi chi tiết hơn về cách dùng mỗi thuật toán này trong các bài tiếp theo.


Về Facenet xem rõ hơn trong Blog này: https://tiensu.github.io/blog/54_face_recognition_facenet/

Về VGGFace xem rõ hơn trong Blog này: https://tiensu.github.io/blog/53_face_recognition_vggface/

Face Classification

Nhiệm vụ của bước này là phân loại khuôn mặt vào các nhóm xác định trước trong tập dữ liệu, dựa vào Vector đặc trưng của chúng. Chúng ta có 3 phương pháp:

    Dựa vào khoảng cách: Tính toán và so sánh khoảng cách giữa các Vectors. Khoảng cách càng nhỏ chứng tỏ các Vectors càng giống nhau. Thuật toán kNN là đại diện tiêu biểu cho việc sử dụng khoảng cách để phân loại, ta có thể áp dụng nó. Khoảng cách ở đây có thể sử dụng công thức Cosine hoặc Euclidean. Ưu điểm của phương pháp này là đơn giản, thực thi nhanh nếu số lượng khuôn mặt không nhiều. Nhược điểm là độ chính xác không cao, tốc độ thực thi giảm nếu số lượng khuôn mặt tăng lên.
    Sử dụng ML: Ta có thể dùng các Vectors đặc trưng của khuôn mặt để huấn luyện một ML model, với các thuật toán như SVM, Decision Tree, … Thuật toán SVM thường được sử dụng nhiều hơn. Phương pháp này cân bằng giữa tốc độ thực hiện và độ chính xác.
    Sử dụng DL: Tương tự vậy, ta cũng có thể huấn luyện một DL model đơn giản (3-5 FC layers) từ các Vectors đặc trưng của khuôn mặt. Phương pháp này thường có độ chính xác cao nhất (nếu DL model đủ tốt), nhưng tốc độ thực hiện lại chậm nhất.

Ngoài ra, trong các bài toán thực tế, để tăng độ chính xác lên cao nhất có thể, chúng ta có thể kết hợp phương pháp 1 và 3, hoặc phương pháp 1 và 2.


# Code

## Data collection

## Data preprocessing

## Device dataset

ta chia dataset ra lam 3 phan: training, validation, testing . validation dung de tuning hyperparameter, su do, ta gop training va validation lai de train model cuoi cung, testing set dung de kiem tra model cuoi cung.


## Pre processing

Vấn đề: 

Hầu hết các thuật toán nhận dạng khuôn mặt cực kỳ nhạy cảm với điều kiện ánh sáng, do đó nếu nó được huấn luyện để nhận ra một người khi họ đang ở trong một căn phòng tối, thì có lẽ nó wont nhận ra họ trong một căn phòng sáng, vv Vấn đề này được gọi là "lumination phụ thuộc ",

cũng có nhiều vấn đề khác, chẳng hạn như mặt cũng phải ở trong một vị trí rất phù hợp trong những hình ảnh (như mắt là trong cùng một điểm ảnh tọa độ), kích thước phù hợp, góc quay, tóc và trang điểm, cảm xúc ( mỉm cười, giận dữ, vv), vị trí của đèn (bên trái hoặc bên trên, vv). 


- .....

Đây là lý do tại sao nó là rất quan trọng để sử dụng một bộ lọc hình ảnh tốt tiền xử lý trước khi áp dụng nhận dạng khuôn mặt. Bạn cũng nên làm những việc như loại bỏ các điểm ảnh xung quanh khuôn mặt mà không được sử dụng, chẳng hạn như với một mặt nạ hình elip để chỉ hiển thị các khu vực mặt bên trong, không phải là tóc và hình nền, kể từ khi họ thay đổi nhiều so với khuôn mặt không. 

Để đơn giản , hệ thống nhận diện khuôn mặt tôi sẽ cho bạn thấy là Eigenfaces sử dụng hình ảnh thang độ xám. Vì vậy, tôi sẽ cho bạn thấy làm thế nào để dễ dàng chuyển đổi hình ảnh màu xám (còn được gọi là "màu xám"), và sau đó dễ dàng áp dụng Histogram Equalization là một phương pháp rất đơn giản của tự động tiêu chuẩn hóa độ sáng và độ tương phản của hình ảnh gương mặt của bạn. Để có kết quả tốt hơn, bạn có thể sử dụng màu sắc nhận diện khuôn mặt (lý tưởng với phụ kiện màu sắc biểu đồ trong HSV hoặc một không gian màu thay vì RGB), hoặc áp dụng nhiều công đoạn chế biến như tăng cường cạnh, phát hiện đường viền, phát hiện chuyển động, vv Ngoài ra, mã này là thay đổi kích thước hình ảnh đến một kích thước tiêu chuẩn, nhưng điều này có thể thay đổi tỷ lệ khía cạnh của khuôn mặt.



## Face detection va Face recognitiojn


Phan 1
https://viblo.asia/p/nhan-dien-khuon-mat-voi-mang-mtcnn-va-facenet-phan-1-Qbq5QDN4lD8


Phan 2
https://viblo.asia/p/nhan-dien-khuon-mat-voi-mang-mtcnn-va-facenet-phan-2-bJzKmrVXZ9N


https://www.pandaml.com/nhan-dang-khuon-mat/

https://medium.com/@shendyeff/documentation-project-face-recognition-system-with-opencv-2d18793623d6


https://www.datacamp.com/tutorial/face-detection-python-opencv



## Processing

https://www.analyticsvidhya.com/blog/2022/04/face-recognition-system-using-python/






Step 1: Face detection

In openCV, the face detection is done by using Haar Cascades (còn gọi là phương pháp Viola-Jones). Haar Cascade is a machine learning object detection algorithm used to identify objects in an image or video and based on the concept of features proposed by Paul Viola and Michael Jones in their paper "Rapid Object Detection using a Boosted Cascade of Simple Features" in 2001. 


Reference: 

https://tiensu.github.io/blog/52_face_recognition/


https://www.analyticsvidhya.com/blog/2022/04/face-recognition-system-using-python/

Du an Github tham khao:

https://github.com/huytranvan2010/Face-Recognition-with-OpenCV-Python-DL