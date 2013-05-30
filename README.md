//including the header files
#include "cv.h"
#include "highgui.h"
#include "math.h"
#include "cxcore.h"

//variable declarations
static CvMemStorage* storage = 0;
static CvHaarClassifierCascade* cascade = 0;
static CvMemStorage* storage1 = 0;
static CvHaarClassifierCascade* cascade1 = 0;

const char* cascade_name = "C:/OpenCV2.1/data/haarcascades/haarcascade_frontalface_alt.xml"; 
const char* cascade_name1 = "C:/OpenCV2.1/data/haarcascades/haarcascade_eye.xml"; 


//main function
int main()
{          
     //declare variables for image
     IplImage * img;
     IplImage * input;
     char  *nameim="face1.jpg";
     
     //get input from camera (input)
     input = cvLoadImage(nameim, CV_LOAD_IMAGE_UNCHANGED);
     
     //initialise img to the size of input
     img = cvCreateImage(cvGetSize(input),IPL_DEPTH_8U,1);
     
     //convert input to Greyscale and store in image
     cvCvtColor(input,img,CV_RGB2GRAY);
     
     //variables to crop face
     int face_origin_x,face_origin_y,width,hieght;
       
     //load the face detedction cascade
     cascade = (CvHaarClassifierCascade*)cvLoad( cascade_name, 0, 0, 0 ); 
    
         //Checks if casscade has been loaded succesfully
         if( !cascade )        
           {fprintf(stderr, "ERROR: Could not load classifier cascade\n" );
            return -1;}
     
     //create the storage element
     storage = cvCreateMemStorage(0);
     
     //set the scaling factor.
     int scale = 1; 
     
     //Create two points to represent the face locations.
     CvPoint pt1,pt2;
     int face_number;
         
     //Carries face detection
     CvSeq* faces = cvHaarDetectObjects( img, cascade, storage,1.1, 2, CV_HAAR_DO_CANNY_PRUNING,cvSize(40, 40) ); 
      
     
         cascade1 = (CvHaarClassifierCascade*)cvLoad( cascade_name1, 0, 0, 0 ); 
        
             //Checks if casscade has been loaded succesfully
             if( !cascade1 )        
               {fprintf(stderr, "ERROR: Could not load classifier cascade1\n" );
                return -1;}
         
         //create the storage element
         storage1 = cvCreateMemStorage(0);
         
         //set the scaling factor.
         int scale1 = 1; 
         
         //Create two points to represent the face locations.
         CvPoint pt11,pt12;
         int face_number1;
             
         //Carries face detection
         CvSeq* faces1 = cvHaarDetectObjects( img, cascade1, storage1,1.1, 2, CV_HAAR_DO_CANNY_PRUNING,cvSize(40, 40) ); 
         for( face_number1 = 0; face_number1 < (faces1 ? faces1->total : 0); face_number1++ )
         {
         
         //Declares rectangle to cover face
         CvRect* r1 = (CvRect*)cvGetSeqElem( faces1, face_number1 );  
         
         //Specifies the points for rectangle.
         /* pt1_____________
            |              |
            |              |
            |              |
            |_____________pt2 */
          pt11.x = r1->x*scale1;                       
          pt12.x= (r1->x+r1->width)*scale1;
          pt11.y = r1->y*scale1;
          pt12.y = (r1->y+r1->height)*scale1;   
          
      cvRectangle( input, pt11, pt12, CV_RGB(0,255,255), 1, 8, 0 );  }    
     
     for( face_number = 0; face_number < (faces ? faces->total : 0); face_number++ )
     {
     
     //Declares rectangle to cover face
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
      
     
      cvRectangle( input, pt1, pt2, CV_RGB(255,255,255), 1, 8, 0 );  }
      
               //Name the window that shows the result.
      cvNamedWindow("O-O", 1);
      
      //Show the window with the image.
      cvShowImage("O-O",input);                                       
    
      //Wait till a key is pressed.
      cvWaitKey(0);          
}


