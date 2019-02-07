#include <stdio.h>
#include <algorithm>
#include <math.h>
#include <iostream>
#include <sys/time.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "mtcnn.h"

//#include <opencv2/opencv.hpp>
//#include <opencv2/face.hpp>
//#include <opencv/highgui.h>

//#include "drawLandmarks.hpp"

//using namespace std;
//using namespace cv;
//using namespace cv::face;


//#include <dlib/image_processing/frontal_face_detector.h>
//#include <dlib/gui_widgets.h>
//#include <dlib/image_io.h>
//#include <iostream>

//using namespace dlib;

bool cmpScore(orderScore lsh, orderScore rsh){
    if(lsh.score<rsh.score)
        return true;
    else
        return false;
}

static float getElapse(struct timeval *tv1,struct timeval *tv2)
{
    float t = 0.0f;
    if (tv1->tv_sec == tv2->tv_sec)
        t = (tv2->tv_usec - tv1->tv_usec)/1000.0f;
    else
        t = ((tv2->tv_sec - tv1->tv_sec) * 1000 * 1000 + tv2->tv_usec - tv1->tv_usec)/1000.0f;
    return t;
}

mtcnn::mtcnn(){
    Pnet.load_param("det1.param");
    Pnet.load_model("det1.bin");
    Rnet.load_param("det2.param");
    Rnet.load_model("det2.bin");
    Onet.load_param("det3.param");
    Onet.load_model("det3.bin");
}

void mtcnn::generateBbox(ncnn::Mat score, ncnn::Mat location, std::vector<Bbox>& boundingBox_, std::vector<orderScore>& bboxScore_, float scale){
    int stride = 2;
    int cellsize = 12;
    int count = 0;
    //score p
    float *p = score.channel(1);//score.data + score.cstep;
    float *plocal = location.data;
    Bbox bbox;
    orderScore order;
    for(int row=0;row<score.h;row++){
        for(int col=0;col<score.w;col++){
            if(*p>threshold[0]){
                bbox.score = *p;
                order.score = *p;
                order.oriOrder = count;
                bbox.x1 = round((stride*col+1)/scale);
                bbox.y1 = round((stride*row+1)/scale);
                bbox.x2 = round((stride*col+1+cellsize)/scale);
                bbox.y2 = round((stride*row+1+cellsize)/scale);
                bbox.exist = true;
                bbox.area = (bbox.x2 - bbox.x1)*(bbox.y2 - bbox.y1);
                for(int channel=0;channel<4;channel++)
                    bbox.regreCoord[channel]=location.channel(channel)[0];
                boundingBox_.push_back(bbox);
                bboxScore_.push_back(order);
                count++;
            }
            p++;
            plocal++;
        }
    }
}
void mtcnn::nms(std::vector<Bbox> &boundingBox_, std::vector<orderScore> &bboxScore_, const float overlap_threshold, string modelname){
    if(boundingBox_.empty()){
        return;
    }
    std::vector<int> heros;
    //sort the score
    sort(bboxScore_.begin(), bboxScore_.end(), cmpScore);

    int order = 0;
    float IOU = 0;
    float maxX = 0;
    float maxY = 0;
    float minX = 0;
    float minY = 0;
    while(bboxScore_.size()>0){
        order = bboxScore_.back().oriOrder;
        bboxScore_.pop_back();
        if(order<0)continue;
        if(boundingBox_.at(order).exist == false) continue;
        heros.push_back(order);
        boundingBox_.at(order).exist = false;//delete it

        for(int num=0;num<boundingBox_.size();num++){
            if(boundingBox_.at(num).exist){
                //the iou
                maxX = (boundingBox_.at(num).x1>boundingBox_.at(order).x1)?boundingBox_.at(num).x1:boundingBox_.at(order).x1;
                maxY = (boundingBox_.at(num).y1>boundingBox_.at(order).y1)?boundingBox_.at(num).y1:boundingBox_.at(order).y1;
                minX = (boundingBox_.at(num).x2<boundingBox_.at(order).x2)?boundingBox_.at(num).x2:boundingBox_.at(order).x2;
                minY = (boundingBox_.at(num).y2<boundingBox_.at(order).y2)?boundingBox_.at(num).y2:boundingBox_.at(order).y2;
                //maxX1 and maxY1 reuse 
                maxX = ((minX-maxX+1)>0)?(minX-maxX+1):0;
                maxY = ((minY-maxY+1)>0)?(minY-maxY+1):0;
                //IOU reuse for the area of two bbox
                IOU = maxX * maxY;
                if(!modelname.compare("Union"))
                    IOU = IOU/(boundingBox_.at(num).area + boundingBox_.at(order).area - IOU);
                else if(!modelname.compare("Min")){
                    IOU = IOU/((boundingBox_.at(num).area<boundingBox_.at(order).area)?boundingBox_.at(num).area:boundingBox_.at(order).area);
                }
                if(IOU>overlap_threshold){
                    boundingBox_.at(num).exist=false;
                    for(vector<orderScore>::iterator it=bboxScore_.begin(); it!=bboxScore_.end();it++){
                        if((*it).oriOrder == num) {
                            (*it).oriOrder = -1;
                            break;
                        }
                    }
                }
            }
        }
    }
    for(int i=0;i<heros.size();i++)
        boundingBox_.at(heros.at(i)).exist = true;
}
void mtcnn::refineAndSquareBbox(vector<Bbox> &vecBbox, const int &height, const int &width){
    if(vecBbox.empty()){
        cout<<"Bbox is empty!!"<<endl;
        return;
    }
    float bbw=0, bbh=0, maxSide=0;
    float h = 0, w = 0;
    float x1=0, y1=0, x2=0, y2=0;
    for(vector<Bbox>::iterator it=vecBbox.begin(); it!=vecBbox.end();it++){
        if((*it).exist){
            bbw = (*it).x2 - (*it).x1 + 1;
            bbh = (*it).y2 - (*it).y1 + 1;
            x1 = (*it).x1 + (*it).regreCoord[0]*bbw;
            y1 = (*it).y1 + (*it).regreCoord[1]*bbh;
            x2 = (*it).x2 + (*it).regreCoord[2]*bbw;
            y2 = (*it).y2 + (*it).regreCoord[3]*bbh;

            w = x2 - x1 + 1;
            h = y2 - y1 + 1;
          
            maxSide = (h>w)?h:w;
            x1 = x1 + w*0.5 - maxSide*0.5;
            y1 = y1 + h*0.5 - maxSide*0.5;
            (*it).x2 = round(x1 + maxSide - 1);
            (*it).y2 = round(y1 + maxSide - 1);
            (*it).x1 = round(x1);
            (*it).y1 = round(y1);

            //boundary check
            if((*it).x1<0)(*it).x1=0;
            if((*it).y1<0)(*it).y1=0;
            if((*it).x2>width)(*it).x2 = width - 1;
            if((*it).y2>height)(*it).y2 = height - 1;

            it->area = (it->x2 - it->x1)*(it->y2 - it->y1);
        }
    }
}
void mtcnn::detect(ncnn::Mat& img_, std::vector<Bbox>& finalBbox_){
    firstBbox_.clear();
    firstOrderScore_.clear();
    secondBbox_.clear();
    secondBboxScore_.clear();
    thirdBbox_.clear();
    thirdBboxScore_.clear();
    
    img = img_;
    img_w = img.w;
    img_h = img.h;
    img.substract_mean_normalize(mean_vals, norm_vals);

    float minl = img_w<img_h?img_w:img_h;
    int MIN_DET_SIZE = 12;
    int minsize = 40;
    float m = (float)MIN_DET_SIZE/minsize;
    minl *= m;
    float factor = 0.709;
    int factor_count = 0;
    vector<float> scales_;
    while(minl>MIN_DET_SIZE){
        if(factor_count>0)m = m*factor;
        scales_.push_back(m);
        minl *= factor;
        factor_count++;
    }
    orderScore order;
    int count = 0;

    for (size_t i = 0; i < scales_.size(); i++) {
        int hs = (int)ceil(img_h*scales_[i]);
        int ws = (int)ceil(img_w*scales_[i]);
        //ncnn::Mat in = ncnn::Mat::from_pixels_resize(image_data, ncnn::Mat::PIXEL_RGB2BGR, img_w, img_h, ws, hs);
        ncnn::Mat in;
        resize_bilinear(img_, in, ws, hs);
        //in.substract_mean_normalize(mean_vals, norm_vals);
        ncnn::Extractor ex = Pnet.create_extractor();
        ex.set_light_mode(true);
        ex.input("data", in);
        ncnn::Mat score_, location_;
        ex.extract("prob1", score_);
        ex.extract("conv4-2", location_);
        std::vector<Bbox> boundingBox_;
        std::vector<orderScore> bboxScore_;
        generateBbox(score_, location_, boundingBox_, bboxScore_, scales_[i]);
        nms(boundingBox_, bboxScore_, nms_threshold[0]);

        for(vector<Bbox>::iterator it=boundingBox_.begin(); it!=boundingBox_.end();it++){
            if((*it).exist){
                firstBbox_.push_back(*it);
                order.score = (*it).score;
                order.oriOrder = count;
                firstOrderScore_.push_back(order);
                count++;
            }
        }
        bboxScore_.clear();
        boundingBox_.clear();
    }
    //the first stage's nms
    if(count<1)return;
    nms(firstBbox_, firstOrderScore_, nms_threshold[0]);
    refineAndSquareBbox(firstBbox_, img_h, img_w);
    printf("firstBbox_.size()=%d\n", firstBbox_.size());

    //second stage
    count = 0;
    for(vector<Bbox>::iterator it=firstBbox_.begin(); it!=firstBbox_.end();it++){
        if((*it).exist){
            ncnn::Mat tempIm;
            copy_cut_border(img, tempIm, (*it).y1, img_h-(*it).y2, (*it).x1, img_w-(*it).x2);
            ncnn::Mat in;
            resize_bilinear(tempIm, in, 24, 24);
            ncnn::Extractor ex = Rnet.create_extractor();
            ex.set_light_mode(true);
            ex.input("data", in);
            ncnn::Mat score, bbox;
            ex.extract("prob1", score);
            ex.extract("conv5-2", bbox);
            if(*(score.data+score.cstep)>threshold[1]){
                for(int channel=0;channel<4;channel++)
                    it->regreCoord[channel]=bbox.channel(channel)[0];//*(bbox.data+channel*bbox.cstep);
                it->area = (it->x2 - it->x1)*(it->y2 - it->y1);
                it->score = score.channel(1)[0];//*(score.data+score.cstep);
                secondBbox_.push_back(*it);
                order.score = it->score;
                order.oriOrder = count++;
                secondBboxScore_.push_back(order);
            }
            else{
                (*it).exist=false;
            }
        }
    }
    printf("secondBbox_.size()=%d\n", secondBbox_.size());
    if(count<1)return;
    nms(secondBbox_, secondBboxScore_, nms_threshold[1]);
    refineAndSquareBbox(secondBbox_, img_h, img_w);

    //third stage 
    count = 0;
    for(vector<Bbox>::iterator it=secondBbox_.begin(); it!=secondBbox_.end();it++){
        if((*it).exist){
            ncnn::Mat tempIm;
            copy_cut_border(img, tempIm, (*it).y1, img_h-(*it).y2, (*it).x1, img_w-(*it).x2);
            ncnn::Mat in;
            resize_bilinear(tempIm, in, 48, 48);
            ncnn::Extractor ex = Onet.create_extractor();
            ex.set_light_mode(true);
            ex.input("data", in);
            ncnn::Mat score, bbox, keyPoint;
            ex.extract("prob1", score);
            ex.extract("conv6-2", bbox);
            ex.extract("conv6-3", keyPoint);
            if(score.channel(1)[0]>threshold[2]){
                for(int channel=0;channel<4;channel++)
                    it->regreCoord[channel]=bbox.channel(channel)[0];
                it->area = (it->x2 - it->x1)*(it->y2 - it->y1);
                it->score = score.channel(1)[0];
                for(int num=0;num<5;num++){
                    (it->ppoint)[num] = it->x1 + (it->x2 - it->x1)*keyPoint.channel(num)[0];
                    (it->ppoint)[num+5] = it->y1 + (it->y2 - it->y1)*keyPoint.channel(num+5)[0];
                }

                thirdBbox_.push_back(*it);
                order.score = it->score;
                order.oriOrder = count++;
                thirdBboxScore_.push_back(order);
            }
            else
                (*it).exist=false;
            }
        }

    printf("thirdBbox_.size()=%d\n", thirdBbox_.size());
    if(count<1)return;
    refineAndSquareBbox(thirdBbox_, img_h, img_w);
    nms(thirdBbox_, thirdBboxScore_, nms_threshold[2], "Min");
    finalBbox_ = thirdBbox_;
}

void test_video() {
	std::string model_path = "../models";
	mtcnn mm;
	cv::VideoCapture mVideoCapture(0);
	if (!mVideoCapture.isOpened()) {
		return;
	}
	cv::Mat frame;
	mVideoCapture >> frame;
	while (!frame.empty()) {
		mVideoCapture >> frame;
		if (frame.empty()) {
			break;
		}

		clock_t start_time = clock();

		ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(frame.data, ncnn::Mat::PIXEL_BGR2RGB, frame.cols, frame.rows);
		std::vector<Bbox> finalBbox;
		mm.detect(ncnn_img, finalBbox);
		const int num_box = finalBbox.size();
		std::vector<cv::Rect> bbox;
		bbox.resize(num_box);
		for(int i = 0; i < num_box; i++){
			bbox[i] = cv::Rect(finalBbox[i].x1, finalBbox[i].y1, finalBbox[i].x2 - finalBbox[i].x1 + 1, finalBbox[i].y2 - finalBbox[i].y1 + 1);
		 }
		for (vector<cv::Rect>::iterator it = bbox.begin(); it != bbox.end(); it++) {
			rectangle(frame, (*it), Scalar(0, 0, 255), 2, 8, 0);
		}
		imshow("face_detection", frame);
		clock_t finish_time = clock();
		double total_time = (double)(finish_time - start_time) / CLOCKS_PER_SEC;
		std::cout << "time" << total_time * 1000 << "ms" << std::endl;

		int q = cv::waitKey(10);
		if (q == 27) {
			break;
		}
	}
	return ;
}

int test_picture(){
	std::string model_path = "../models";
	mtcnn mm;

	std::cout << "after load model..." << std::endl;
	clock_t start_time = clock();

	cv::Mat image;
	image = cv::imread("../sample.jpg");
	ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(image.data, ncnn::Mat::PIXEL_BGR2RGB, image.cols, image.rows);
	std::vector<Bbox> finalBbox;
	mm.detect(ncnn_img, finalBbox);

	const int num_box = finalBbox.size();
	cout << "num_box: " << num_box << endl;
	std::vector<cv::Rect> bbox;
	bbox.resize(num_box);
	for (int i = 0; i < num_box; i++) {
		bbox[i] = cv::Rect(finalBbox[i].x1, finalBbox[i].y1, finalBbox[i].x2 - finalBbox[i].x1 + 1, finalBbox[i].y2 - finalBbox[i].y1 + 1);

	}
	for (vector<cv::Rect>::iterator it = bbox.begin(); it != bbox.end(); it++) {
		rectangle(image, (*it), Scalar(0, 0, 255), 2, 8, 0);
	}

	std::cout << "bbox size: " << bbox.size() << std::endl;

	exit(0);
	imshow("face_detection", image);
	clock_t finish_time = clock();
	double total_time = (double)(finish_time - start_time) / CLOCKS_PER_SEC;
	std::cout << "time" << total_time * 1000 << "ms" << std::endl;

	cv::waitKey(0);

}


// get x and y of rectangle or eyes and calc diagonal
// diagonal is the top size
int face_size(int x1, int y1, int x2, int y2)
{
	int x_2 = (x1-x2)*(x1-x2);
	int y_2 = (y1-y2)*(y1-y2);

	int size = sqrt(x_2+y_2);
	return size;
}

// get eyes angle
// or rectangle angle
float face_angle(Mat frame, int x1, int y1, int x2, int y2)
{
	float pi=3.14;

	int x=x2-x1;
	int y=y2-y1;

	// x^2 + y2 = 1
	// angle = arccos ()
	float normal = x*x+y*y;
	
	float normalized_x2 = x*x/normal;
	float normalized_y2 = y*y/normal;

	float alfa = sqrt(normalized_x2)/sqrt(normalized_x2+normalized_y2);
	float angle = acos(alfa);//*180/pi;
	
	//cout<<angle<<endl;
	//circle( frame, Point( x1, y1 ), 1.0, Scalar( 0, 0, 255 ), 1, 1 );
	//circle( frame, Point( x2, y2 ), 1.0, Scalar( 0, 0, 255 ), 1, 1 );
	
	if(y<0)
	{
		angle = angle*(-1);
	}
	
	return angle;
}


class buffer
{
public:
// cirlucar buffer
int min_buffer(int min)
{
	int length = 10;
	
	
	bool open_closed;
	
	int biggest = 10000;
	
	static int min_old=biggest;
	static int old_min_position=0;
	
	static int min_next;
	static int min_new_position=0;
	
	static int next_step=0;
	
	if(min<min_next && next_step==1)
	{
		min_next=min;
		min_new_position=0;
		
		next_step=0;
		std::cout << " 1min= " << min_old << " min= " << min_next << " bool o/c= " << open_closed << std::endl;
	}
	if(min<min_old)
	{
		min_next=biggest;
		
		min_old=min;
		old_min_position=0;
		
		next_step=1;
		std::cout << " 2min= " << min_old << " min= " << min_next << " bool o/c= " << open_closed << std::endl;
	}
	
	min_new_position+=1;
	old_min_position+=1;
	if(old_min_position==length)
	{	
		min_old=min_next;
		old_min_position=min_new_position;
		min_next=min;
		std::cout << " 3min= " << min_old << " min= " << min_next << " bool o/c= " << open_closed << std::endl;
	}
	
	//std::cout << " min= " << min_old << " min= " << min_next << " bool o/c= " << open_closed << std::endl;
	
	return min_old;
}
};


// find the pupil on region
int min_max(Mat frame, int x1, int y1, float angle, int sx=5, int sy=2)
{
	
	int min = 255+255+255;
	int x_min =0;
	int y_min =0;

	int max = 0;
	int x_max =0;
	int y_max =0;
		    
	int l_white=0;
	int r_white=0;

/*        sx
    -------------
    |    ---    |
    |  |  0  |  | sy
    |    ---    |
    -------------
*/

	// 2D params of eye:
	// S of region
	// S = x*y

	// min max in rectangle does'n works	
	// find min and max in rectangle
//	for(int y=y1-sy; y<y1+2*sy; y++)  
//	{
//		for(int i=(x1-sx)*3; i<(x1+sx)*3; i=i+3)  
//		{
//			//cout<<(int)cv_img.data[cv_img.step*y+i]<<endl;
//	
//			int gray = frame.data[frame.step*y+i]+frame.data[frame.step*y+i+1]+frame.data[frame.step*y+i+2];
//			
//			// if point is lesser - that is pupil
//			if(min>gray)
//			{
//				min=gray;
//				x_min=i;
//				y_min=y;
//			}
//	
//			
//	
//			// if point is higher - that is pupil
//			if(max<gray)
//			{
//				max=gray;
//				x_max=i;
//				y_max=y;
//			}
//		}
//	}
	
	// ------------------------------------
	// left or right
	
	
	int pix=sx/5;
	int sum_l=0;
	int sum_r=0;

	int j=y1;
	float jj=y1;


// get line point's sum and local min/max
// 

	int local_x;
	int local_y;
	int minn=255+255+255;

int s=j;

for(int j=s-3; j<s+3; j++)  
{
	for(int i=x1; i>x1-pix; i--)  
	{
		//get the minimum of region and check for open/closed

		int io=i;
		int jo=j;
		sum_l=frame.data[frame.step*jo+io*3+0]+frame.data[frame.step*jo+io*3+1]+frame.data[frame.step*jo+io*3+2];
		io=i+1;
		sum_l=frame.data[frame.step*jo+io*3+0]+frame.data[frame.step*jo+io*3+1]+frame.data[frame.step*jo+io*3+2];
		jo=j+1;
		sum_l=frame.data[frame.step*jo+io*3+0]+frame.data[frame.step*jo+io*3+1]+frame.data[frame.step*jo+io*3+2];
		jo=j-1;
		sum_l=frame.data[frame.step*jo+io*3+0]+frame.data[frame.step*jo+io*3+1]+frame.data[frame.step*jo+io*3+2];
		io=i-1;

		if(sum_l<minn)
		{	
			minn=sum_l;
			local_y=y1;
			local_x=i;
		}  // get the min of local

		//cout << " angle " << jj << endl;
		jj-=angle*1.5;//cos(angle);
		jo=jj;

		// debug
		frame.data[frame.step*jo+io*3+0]=255;
		frame.data[frame.step*jo+io*3+1]=255;
		frame.data[frame.step*jo+io*3+2]=255;
	}
}

		
	// if eye is closed - color is higher
	//if(min_new*1.1<minn)
	//	{cout << "closed"<<"min_new" << min_new << " min" << minn << endl;}
	//cout << "closed"<<"min_new" << min_new << " min" << minn << endl;
	
	jj=y1;
	j=y1;
	for(int i=x1; i<x1+pix; i++)  
	{


		jj+=angle*1.5;//cos(angle);
		j=jj;

		int io=i;
		int jo=j;
		sum_l=frame.data[frame.step*jo+io*3+0]+frame.data[frame.step*jo+io*3+1]+frame.data[frame.step*jo+io*3+2];
		io=i+1;
		sum_l=frame.data[frame.step*jo+io*3+0]+frame.data[frame.step*jo+io*3+1]+frame.data[frame.step*jo+io*3+2];
		jo=j+1;
		sum_l=frame.data[frame.step*jo+io*3+0]+frame.data[frame.step*jo+io*3+1]+frame.data[frame.step*jo+io*3+2];
		jo=j-1;
		sum_l=frame.data[frame.step*jo+io*3+0]+frame.data[frame.step*jo+io*3+1]+frame.data[frame.step*jo+io*3+2];
		io=i-1;

		if(sum_l<minn)
		{	
			minn=sum_l;
			local_y=y1;
			local_x=i;
		}  // get the min of local

		//debug
		frame.data[frame.step*j+i*3+0]=255;
		frame.data[frame.step*j+i*3+1]=255;
		frame.data[frame.step*j+i*3+2]=255;

	}

	

	// open closed buffer 
	buffer left;
	buffer right;

	static int min_new = left.min_buffer(minn);
	if(min_new < 1.1*(float)minn)
		{
			float out_angle = (x1 - local_x)*20/pix;
			return out_angle;
		}

	circle( frame, Point( local_x, local_y ), 1, Scalar( 255, 0, 0 ), 1, 1 );

	//cout << "pupil " << frame.data[frame.step*y1+x1*3+0] + frame.data[frame.step*y1+x1*3+1] + frame.data[frame.step*y1+x1*3+2];
	
	// ------------------------------------
	//cout<<"left_right"<<sum_l<<" "<<sum_r<<endl;

	//debug
	
	//int white = frame.data[frame.step*y_max+x_max]+frame.data[frame.step*y_max+x_max+1]+frame.data[frame.step*y_max+x_max+2];
	//int dark  = frame.data[frame.step*y_min+x_min]+frame.data[frame.step*y_min+x_min+1]+frame.data[frame.step*y_min+x_min+2];
	
	//circle( frame, Point( x_min/3, y_min ), 0.3, Scalar( 0, 0, 255 ), 1, 1 );
	//circle( frame, Point( x_min/3, y_min ), 0.3, Scalar( 0, 0, 255 ), 1, 1 );
	
	
	//cout << "wd = " <<white << " " << dark << endl;
	
	//line( frame, Point( (x1+sx), y1+2*sy ),Point( (x1-sx), y1-sy ),  Scalar( 0, 0, 255 ));
	
	// left or right
	//cout << 
	
	return 0;
}


bool looks(Mat frame,int size,float angle,int x1,int y1,int x2,int y2)
{
	int j=0;
	for(int i=x1; i<size; i++)
	{
		int sum_l=frame.data[frame.step*j+i*3+0]+frame.data[frame.step*j+i*3+1]+frame.data[frame.step*j+i*3+2];

		frame.data[frame.step*j+i*3+0]=255;
		frame.data[frame.step*j+i*3+1]=255;
		frame.data[frame.step*j+i*3+2]=255;
	}

	return 1;
}

int main(int argc, char** argv)
{
    const char* imagepath = argv[1];

    //VideoCapture cap(0);
  //VideoCapture cap("http://192.168.0.15:4747/video");
  VideoCapture cap(imagepath);

    cap.set(CV_CAP_PROP_FRAME_WIDTH, 320);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 240);

  

    

    if(!cap.isOpened())
    {
	printf("not opened %s",imagepath);
	return -1;
    }

    cv::Mat cv_img; //= cv::imread(imagepath, CV_LOAD_IMAGE_COLOR);

//landmarks by lbfmodel
//Ptr<Facemark> facemark = FacemarkLBF::create();

    
    while(1)
    {
            cap >> cv_img;
    	    
	    if (cv_img.empty())
	    {
		fprintf(stderr, "cv::imread %s failed\n", imagepath);
		return -1;
	    }



	    std::vector<Bbox> finalBbox;
	    mtcnn mm;
	    ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(cv_img.data, ncnn::Mat::PIXEL_BGR2RGB, cv_img.cols, cv_img.rows);
	    struct timeval  tv1,tv2;
	    struct timezone tz1,tz2;
	    gettimeofday(&tv1,&tz1);
	    mm.detect(ncnn_img, finalBbox);
	    gettimeofday(&tv2,&tz2);
	    printf( "%s = %g ms \n ", "Detection All time", getElapse(&tv1, &tv2) );

	    int total = 0;

//landmarks by lbfmodel
	

//  Landmarks by mtcnn
	    for(vector<Bbox>::iterator it=finalBbox.begin(); it!=finalBbox.end();it++){
		if((*it).exist){
		    total++;
		    cv::rectangle(cv_img, Point((*it).x1, (*it).y1), Point((*it).x2, (*it).y2), Scalar(0,0,255), 2,8,0);
		    //for(int num=0;num<5;num++)circle(cv_img,Point((int)*(it->ppoint+num), (int)*(it->ppoint+num+5)),3,Scalar(0,255,255), -1);


		    /////////////////////////////////// look or not

		    // left eye
		    int x1=(int)*(it->ppoint+0);
		    int y1=(int)*(it->ppoint+0+5);
		    
		    // right eye
		    int x2=(int)*(it->ppoint+1);
		    int y2=(int)*(it->ppoint+1+5);

		    // nose
		    int x3=(int)*(it->ppoint+2);
		    int y3=(int)*(it->ppoint+2+5);

		    float angle = face_angle(cv_img, x1,y1,x2,y2);
		    int size = face_size(x1, y1, x2, y2);
	

		    // new function with face and id
		    if ( looks(cv_img,size,angle,x1,y1,x2,y2) )
		    {
			char string[128];
			sprintf(string, " look %d", 1);
			putText(cv_img, string, cvPoint((*it).x1 - 5, (*it).y1 - 5), CV_FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1, 8);
		    }



	// oblast
	// bilki
	// nose vs eyes	

		    //int r = min_max(cv_img, x1, y1, angle, size);
		//cout<<" r "<< r << endl;
		    //int l = min_max(cv_img, x2, y2, angle, size);
		//cout<<" l "<< l << endl;
		//cout<<" nose "<< ((*it).x1+(*it).x2)/2 - x3 << endl;

			cv::waitKey(0);

		}
	    }


	    std::cout << "totol detect " << total << " persons" << std::endl;

	    imshow("face_detection", cv_img);
	    //cv::waitKey(0);
	    if(cv::waitKey(1) >=0) break;

	    //cv::imwrite("result.jpg",cv_img);
    }

    return 0;
}
