using System;
using System.Linq;
using System.Runtime.InteropServices;
using OpenCvSharp;
using Numpy;
using Numpy.Models;
using Python.Included;
using Python.Runtime;
using OpenCvSharp.Util;
using System.Diagnostics;

namespace openvtest
{
    class Program
    {
      
        static void Main(string[] args)
        {
            string filePath = @"SampleImage.jpg";
            var img = new Mat(filePath, ImreadModes.Grayscale);

            Mat orginalImg = new Mat();
            img.CopyTo(orginalImg);
            img  = ReSizeImage(img);


            Mat edges = DetectEdges(img);
            var points = GetContours(edges);

            Mat warpedImg = PointTransform(img, points);
            warpedImg = warpedImg.CvtColor(ColorConversionCodes.BayerBG2GRAY);

            var thresh = warpedImg.AdaptiveThreshold(225, AdaptiveThresholdTypes.GaussianC, ThresholdTypes.Binary, 11, 2);
            Cv2.ImShow("results", thresh);
            Cv2.WaitKey();
        }

        private static Mat ReSizeImage(Mat img)
        {
            Size imgsize = img.Size();
            int scale_percent = 50;
            int width = (imgsize.Width * scale_percent / 100);
            int height = (imgsize.Height * scale_percent / 100);

            var resized = new Size(width, height);
           img  = img.Resize(resized);
            return img;
        }

        public static  Mat DetectEdges(Mat src) 
        {
           var dst = new Mat();
            Cv2.Canny(src, dst, 50, 200);
            return dst;
            
        }
        public static Point[] GetContours(Mat edgesImg) {
            Point[] screenCut = null;
            Point[][] contours;
            Mat heirchy = Mat.Zeros(edgesImg.Size(), edgesImg.Type());
            HierarchyIndex[] hIndex;
            Cv2.FindContours(edgesImg,out contours,out hIndex, RetrievalModes.List, ContourApproximationModes.ApproxSimple);

            var query = contours
                 .OrderBy(n => Cv2.ContourArea(n))
                 .Reverse();

            foreach (var contour in query)
            {

                double peri = Cv2.ArcLength(contour, true);
                Point[] approx = Cv2.ApproxPolyDP(contour, 0.02 * peri, true);

                if (approx.Length == 4)
                {
                   screenCut = approx;
                    break;
                }

            }
            return screenCut;
                
           
        }
        private static NDarray OrderPoints(Point[] points)
        {
            int[,]Pts = new int[,]
            {
              {points[0].X, points[0].Y },
              {points[1].X, points[1].Y },
              {points[2].X, points[2].Y } ,
              {points[3].X, points[3].Y }
            };
            NDarray rectangle = null;
            try
            {
                Dtype dtype = Pts.GetDtype();
            NDarray pts = np.array(Pts);
                 rectangle = np.zeros( new Shape(4,2), dtype);

         

            NDarray sum = np.sum(pts, 1);
            NDarray differnce = np.diff(pts, axis: 1);

            rectangle[0] = pts[sum.argmin()];
            rectangle[2] = pts[sum.argmax()];
            rectangle[1] = pts[differnce.argmin()];
            rectangle[3] = pts[differnce.argmax()];
            }
            catch (System.IO.IOException e)
            {

                Debug.Write("Debug Informations");
                Debug.WriteLine(e.Message);
                Debug.WriteLine(e.HResult);
                Debug.WriteLine(e.Data);
            }
            

            return rectangle;
        }
        private static Mat PointTransform(Mat image,Point[] pts)
        {
            int maxWidth, maxHeight;
            NDarray rect;
            GetDimensions(pts, out maxWidth, out maxHeight , out rect);

            // make a top - down view
            var dst = np.array(
                  new[,] {
                     { 0,0},
                     {maxWidth - 1, 0 },
                     {maxWidth - 1, maxHeight - 1 },
                     {0, maxHeight - 1 }
                   });

            Point2f[] orderedPoints = new Point2f[4];
            Point2f[] dstPts = new Point2f[4];

            for (int c = 0; c < rect.len ; c++)
            {
                int[] data = rect[c].GetData<int>();
                orderedPoints[c] = new Point2f(data[0],data[1]);
            }

            for (int i = 0; i < dst.len; i++)
            {
                int[] data = dst[i].GetData<int>();
                dstPts[i] = new Point2f(data[0], data[1]);
            }


            Mat matr = Cv2.GetPerspectiveTransform(orderedPoints, dstPts);
            Mat output = new Mat();
            Cv2.WarpPerspective(image, output, matr, new Size(maxWidth, maxHeight));

            return output;
        }
        private static void GetDimensions(Point[] pts, out int maxWidth, out int maxHeight, out NDarray rect)
        {
             rect = OrderPoints(pts);
           
            NDarray tL = rect[0], tR = rect[1], bR = rect[2], bL = rect[3];

            
            NDarray w1 = ((bR[0] - bL[0]) * (bR[0] - bL[0])) + ((bR[1] - bL[1]) * (bR[1] - bL[1]));
            NDarray w2 = ((tR[0] - tL[0]) * (tR[0] - tL[0])) + ((tR[1] - tL[1]) * (tR[1] - tL[1]));

            w1 = np.sqrt(w1);
            w2 = np.sqrt(w2);


            int width1 = (int)float.Parse(w1.repr);
            int width2 = (int)float.Parse(w2.repr);

            maxWidth = Math.Max(width1, width2);


            NDarray h1 = ((tR[0] - bR[0]) * (tR[0] - bR[0])) + ((tR[1] - bR[1]) * (tR[1] - bR[1]));
            NDarray h2 = (tL[0] - bL[0]) * (tL[0] - bL[0]) + (tL[1] - bL[1]) * (tL[1] - bL[1]);

            h1 = np.sqrt(h1);
            h2 = np.sqrt(h2);

            int height1 = (int)float.Parse(h1.repr);
            int height2 = (int)float.Parse(h2.repr);


            maxHeight = Math.Max(height1, height2);
        }
    }
}
