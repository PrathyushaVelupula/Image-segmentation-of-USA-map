import argparse
import cv2
import numpy as np

class ImageSegment():
    def __init__(self, input_image, clusters, waterbodydetection) -> None:
        self.input_image = cv2.imread('usa.png')
        
        self.input_image = cv2.copyMakeBorder(self.input_image.copy(), 100, 100, 100, 100,cv2.BORDER_CONSTANT,value=(255, 255, 255))
        self.clusters = clusters
        self.waterbodydetection = waterbodydetection
  
    def segment_image(self):
        image = cv2.cvtColor(self.input_image, cv2.COLOR_BGR2RGB)
        pixel_values = image.reshape((-1, 3))
        pixel_values = np.float32(pixel_values)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, (centers) = cv2.kmeans(pixel_values, self.clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers) 
        labels = labels.flatten()
        segmented_image = centers[labels.flatten()]
        segmented_image = segmented_image.reshape(image.shape)
        return segmented_image, labels
    
    def image_segmentation(self):
        # Remove white background
        image = self.input_image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        image[thresh == 255] = 0
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # erosion to remove any remaining small white blobs or noise in the image    
        self.input_image = cv2.erode(image, kernel, iterations = 1)
        image = cv2.cvtColor(self.input_image, cv2.COLOR_BGR2RGB)
        pixel_values = image.reshape((-1, 3))  
        pixel_values = np.float32(pixel_values)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, (centers) = cv2.kmeans(pixel_values, self.clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        labels = labels.flatten()
        segmented_image = centers[labels.flatten()]
        segmented_image = segmented_image.reshape(image.shape)


        all_countours = [] #returns the list of all contours and the dictionary of cluster labels and their respective contours        
        all_countours_dict = {} #list of contours for each cluster label
        
        for cluster_label in range(self.clusters):
            masked_image = segmented_image.copy()            
            masked_image = masked_image.reshape((-1, 3))
            masked_image[labels != cluster_label] = [0, 0, 0]           
            masked_image = masked_image.reshape(segmented_image.shape)
            # convert to grayscale because findContours takes grayscale image as input
            masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
            _, edges =  cv2.threshold(masked_image, 20, 255, cv2.THRESH_BINARY)
            #external contours of each connected component in the binary image are returned
            #endpoints of the contour segments are stored
            contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                all_countours_dict[cluster_label] = contours
                all_countours.extend(contours)

        # Define the callback function to handle mouse events
        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if self.waterbodydetection:
                    selected_label = None
                    for label in all_countours_dict:
                        contours = all_countours_dict[label]
                        is_point_in_countour = np.argmax([cv2.pointPolygonTest(contour, (x, y), False) for contour in contours])
                        contour = contours[is_point_in_countour]
                        #check if the point is exactly on the contour(1) or if it is very close to the contour (0) 
                        if cv2.pointPolygonTest(contour, (x, y), False) == 1 or cv2.pointPolygonTest(contour, (x, y), False) == 0:
                            selected_label = label
                    contour = []
                    state_contour = []
                    distances = [cv2.pointPolygonTest(contour, (x, y), True) for contour in all_countours_dict[selected_label]]
                    for i in range(0, len(distances)):
                        initial_distance = np.argmax(distances)
                        next_distance = abs(distances[i])
                        diff_distance = abs(initial_distance - next_distance)
                        if diff_distance < 200 :
                            contour.extend(all_countours_dict[selected_label][i])
                            state_contour.append(all_countours_dict[selected_label][i])
                    contour = np.array(contour)
                else:
                    is_point_in_countour = np.argmax([cv2.pointPolygonTest(contour, (x, y), False) for contour in all_countours])
                    contour = all_countours[is_point_in_countour]
                    print(contour)

                _, _, w, h = cv2.boundingRect(contour)
                color = np.array(self.input_image[y, x], dtype=int).tolist()
                blank = np.zeros(self.input_image.shape, dtype='uint8')
                mask = np.zeros(self.input_image.shape, dtype='uint8')
                cv2.drawContours(blank, state_contour, -1, color, -1)
                cv2.drawContours(mask, state_contour, -1, (255, 255, 255), -1)
                
                # Build Masks
                (y, x, _) = np.where(mask == 255)
                (topy, topx) = (np.min(y), np.min(x))
                (bottomy, bottomx) = (np.max(y), np.max(x))
                state = blank[topy:bottomy+1, topx:bottomx+1]

                # Build resized masks
                resized_state = cv2.resize(state, None, fx=2, fy=2)

                # Get centroids
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"]) if M["m00"] != 0 else 0
                    cY = int(M["m01"] / M["m00"]) if M["m00"] != 0 else 0
                pt1 = (cX, cY)

                pt2 = (resized_state.shape[1] //2, resized_state.shape[0] //2)
                dx = (pt1[0] - pt2[0])
                dy = (pt1[1] - pt2[1])

                h, w = resized_state.shape[:2]
                zoomed_image = np.zeros_like(self.input_image)
                zoomed_image[max(0, dy):min(self.input_image.shape[0], dy + h),
                max(0, dx):min(self.input_image.shape[1], dx + w)] = resized_state[
                                                                   max(0, -dy):min(h, self.input_image.shape[0] - dy),
                                                                   max(0, -dx):min(w, self.input_image.shape[1] - dx)]
                
                roi = self.input_image[0: self.input_image.shape[0], 0: self.input_image.shape[1]]
                zoomed_image_gray = cv2.cvtColor(zoomed_image, cv2.COLOR_BGR2GRAY)
                _, zoomed_image_mask = cv2.threshold(zoomed_image_gray, 20, 255, cv2.THRESH_BINARY)
                #zoomed_image_mask = cv2.dilate(zoomed_image_mask, kernel, iterations=1) 
                zoomed_image_fg = cv2.bitwise_and(zoomed_image, zoomed_image, mask = zoomed_image_mask)
                zoomed_image_mask_inv = cv2.bitwise_not(zoomed_image_mask)
                image_bg = cv2.bitwise_and(roi, roi, mask = zoomed_image_mask_inv)
                dst = cv2.add(image_bg, zoomed_image_fg)
                #dst=cv2.addWeighted(image_bg, 0.8, zoomed_image_fg, 1, 0)
                cv2.namedWindow("Result",cv2.WINDOW_NORMAL)
                cv2.imshow('Result', dst)
        while True:
            cv2.namedWindow("image",cv2.WINDOW_NORMAL)
            cv2.imshow("image", self.input_image)
            cv2.setMouseCallback("image", click_event)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Map Segmentation")
    parser.add_argument("--input_image", help="input image file")
    parser.add_argument('--clusters', type=int, default=10, help='Number of clusters for k-means clustering')
    parser.add_argument('--nowaterbodydetection', action='store_true', help='Detect Water Bodies')
    args = parser.parse_args()
    mapsegment = ImageSegment(args.input_image, args.clusters, not args.nowaterbodydetection)
    mapsegment.image_segmentation()
    cv2.destroyAllWindows()
