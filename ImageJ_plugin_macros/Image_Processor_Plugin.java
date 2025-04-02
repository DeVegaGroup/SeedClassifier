import ij.*;
import ij.plugin.*;
import ij.process.*;
import ij.plugin.filter.*;
import ij.plugin.frame.*;

public class Image_Processor_Plugin implements PlugIn {
    public void run(String arg) {

        // 1. Duplicate the image
        ImagePlus originalImage = IJ.getImage();
        ImagePlus duplicateImage = new Duplicator().run(originalImage);

        // 2. Convert to RGB Stack
        ImageConverter ic = new ImageConverter(duplicateImage);

       ic.convertToRGB();
       duplicateImage.updateAndDraw();

        // 3. Convert image to 8-bit
        ImageConverter id = new ImageConverter(duplicateImage);
        
        id.convertToGray8();
        duplicateImage.updateAndDraw();

        // 4. Apply "Find Edges" on stack
        IJ.run(duplicateImage, "Find Edges", "stack");

        // 5. Set and apply Otsu's auto-threshold method
        IJ.setAutoThreshold(duplicateImage, "Otsu dark");
        IJ.run(duplicateImage, "Convert to Mask", "method=Otsu background=Dark calculate black");

        // 6. Create a mask of the thresholded image
        ImagePlus maskImage = new ImagePlus("Mask", duplicateImage.getProcessor());

        // 7. Apply "dilate"
        IJ.run(maskImage, "Close-", "");
        IJ.run(maskImage, "Dilate", "");

        // 8. Apply "Close-"
    
        IJ.run(maskImage, "Close-", "");

        // 9. Apply "Outline"
        IJ.run(maskImage, "Outline", "");
 
        // 10. Apply "dilate"
        // IJ.run(maskImage, "Close-", "");
        // IJ.run(maskImage, "Dilate", "");

        // 11. Apply "Close-"
    
        // IJ.run(maskImage, "Close-", "");

        // 12. Apply "Outline"
        IJ.run(maskImage, "Outline", "");
        
        // 13. Apply "dilate"
        IJ.run(maskImage, "Close-", "");
        IJ.run(maskImage, "Dilate", "");

        // 14. Apply "Close-"
    
        IJ.run(maskImage, "Close-", "");

        // 15. Apply "Outline"
        IJ.run(maskImage, "Outline", "");

        // 16. Apply "Fill Holes"
        IJ.run(maskImage, "Fill Holes", "");
        // IJ.run(maskImage, "Watershed", "");
        IJ.run(maskImage, "Erode", "");
        IJ.run(maskImage, "Erode", "");
        IJ.run(maskImage, "Erode", "");
        IJ.run(maskImage, "Erode", "");
        IJ.run(maskImage, "Erode", "");

    
       

        // 17. Create selection
        IJ.run(maskImage, "Create Selection", "");

        // 18. Restore Selection to original image
        originalImage.setRoi(maskImage.getRoi());

        // show the result
        originalImage.show();
    }
}

