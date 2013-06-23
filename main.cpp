//including the header files
#include "cv.h"
#include "highgui.h"
#include "math.h"
#include "cxcore.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>



//variable declarations
static CvMemStorage* storage = 0;
static CvHaarClassifierCascade* cascade = 0;
const char* cascade_name = "../project/data/haarcascades/haarcascade_frontalface_alt.xml"; 
const char* cascade_name_eye = "../project/data/haarcascades/haarcascade_eye.xml"; 


void face_detect_crop(IplImage * input,IplImage * output)
{

     IplImage * img;
     img = cvCreateImage(cvGetSize(input),IPL_DEPTH_8U,1);
     cvCvtColor(input,img,CV_RGB2GRAY);//convert input to Greyscale and store in image
     int face_origin_x,face_origin_y,width,hieght;//variables to crop face
     
       
     cascade = (CvHaarClassifierCascade*)cvLoad( cascade_name, 0, 0, 0 ); //load the face detedction cascade
     storage = cvCreateMemStorage(0);
     int scale = 1; 
     CvPoint pt1,pt2;
     int face_number;
         
     CvSeq* faces = cvHaarDetectObjects( img, cascade, storage,1.1, 2, CV_HAAR_DO_CANNY_PRUNING,cvSize(40, 40) ); 
     for( face_number = 0; face_number < (faces ? faces->total : 0); face_number++ )
     {
     	 CvRect* r = (CvRect*)cvGetSeqElem( faces, face_number );  
     
     //Specifies the points for rectangle.
     /* pt1_____________

        |              |

        |              |

        |              |

        |_____________pt2 */
      	pt1.x = r->x*scale;                       
      	pt2.x = (r->x+r->width)*scale;
      	pt1.y = r->y*scale;
      	pt2.y = (r->y+r->height)*scale;   
        cvRectangle( input, pt1, pt2, CV_RGB(255,255,255), 1, 8, 0 );  
        CvRect rs=*r;
      //cvNamedWindow("i-O", 1);
      //cvShowImage("i-O",input);                                       
      //cvWaitKey(0);
      cvSetImageROI(img,rs);
      }
      IplImage * frame;
      CvSize s1={48,48};
      frame=cvCreateImage(s1,IPL_DEPTH_8U,1);
	
      cvResize(img,frame);
      cvCvtColor(frame,output,CV_GRAY2RGB);
      
      CvPoint pt;
      cascade = (CvHaarClassifierCascade*)cvLoad( cascade_name_eye, 0, 0, 0 ); //load the face detedction cascade
      CvSeq* faces1 = cvHaarDetectObjects( input, cascade, storage,1.1, 2, CV_HAAR_DO_CANNY_PRUNING,cvSize(40, 40) ); 
      for( face_number = 0; face_number < (faces1 ? faces1->total : 0); face_number++ )
      {
     	 CvRect* r = (CvRect*)cvGetSeqElem( faces1, face_number );  
      	pt.x = (r->x*scale);                       
      	pt2.x = ((r->x+r->width)*scale);
      	pt.y = (r->y*scale);
      	pt2.y = ((r->y+r->height)*scale);   
        cvRectangle( input, pt, pt2, CV_RGB(0,255,255), 1, 8, 0 );  
      }
      
              
      
}
//main function
int main()
{          
     //declare variables for image
     IplImage * input1;
     IplImage * output1; 
     char  nameim[100]="../project/dbase/males/1/mvc-001f.jpg";
     CvSize s1={48,48};
     output1=cvCreateImage(s1,IPL_DEPTH_8U,3);
	
     CvSVM SVM;
     float a;
     SVM.load("../project/temp/SVM_hap_neu_sad.txt");
     
     FILE *fp;
     float feat[18432];
     char str[50]="./gabor ../project/temp/temp1.jpg ";
             
     IplImage * happy;
     IplImage * sad; 
     IplImage * neutral;
     IplImage * temp;
     CvSize s2={400,400};
     happy=cvCreateImage(s2,IPL_DEPTH_8U,3);
     sad=cvCreateImage(s2,IPL_DEPTH_8U,3);
     neutral=cvCreateImage(s2,IPL_DEPTH_8U,3);
     temp = cvLoadImage("../project/data/Images/happy.jpeg", CV_LOAD_IMAGE_UNCHANGED);
     cvResize(temp,happy);
     temp = cvLoadImage("../project/data/Images/sad.jpeg", CV_LOAD_IMAGE_UNCHANGED);
     cvResize(temp,sad);
     temp = cvLoadImage("../project/data/Images/neutral.jpeg", CV_LOAD_IMAGE_UNCHANGED);
     cvResize(temp,neutral);
            		

     CvCapture *capture=cvCreateCameraCapture(0);
     if(capture!=NULL)  //camera has begun starting itself
     for(;;)
     {
        
	      input1=cvQueryFrame(capture);//take current image in camera and give it to input pointer
	       
	      //get input from camera (input)
	      //input1 = cvLoadImage(nameim, CV_LOAD_IMAGE_UNCHANGED);
	      face_detect_crop(input1,output1);
	      cvSaveImage("../project/temp/temp1.jpg",output1);
	      
//_______________________________________________________________//  

      		fp=popen(str,"r");
      		for(int i=0;i<18432;i++)
      		{
        		fscanf(fp,"%f",&feat[i]);
        		//std::cout<<feat[i]<<" ";
      		}
      		pclose(fp);
     
//_______________________________________________________________//  

	      cvNamedWindow("Emotion", 1);
	      
              cv::Mat testmat(1, 18432, CV_32FC1, feat);
	      a=SVM.predict(testmat);
	      if( a<1.1 && a>0.9)
	      {std::cout<<"happy\n";cvShowImage("Emotion",happy);if( cv::waitKey( 10 ) >= 0 )break;}	      
		else if(a>-1.1 && a<-0.9)   
		{std::cout<<"sad\n";cvShowImage("Emotion",sad);if( cv::waitKey( 10 ) >= 0 )break;}
                else 
		{std::cout<<"neutral\n";cvShowImage("Emotion",neutral);if( cv::waitKey( 10 ) >= 0 )break;}    


		cvNamedWindow("O-O", 1);
	        cvShowImage("O-O",input1);
	        if( cv::waitKey( 10 ) >= 0 )break;
      

     }
     cvReleaseCapture( &capture ); 
     return 0;
}
