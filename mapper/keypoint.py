import cv2 as cv
import numpy as np

import sys

import mapper.image as im


def detect(image: cv.Mat, variant=0):
    assert im.is_image(image), 'Argument is assumed to be an image'

    impl = None

    if variant == 0:  # ORB
        impl = cv.ORB_create()
        impl.setMaxFeatures(10000)
    elif variant == 1:  # AKAZE
        impl = cv.AKAZE_create()
    elif variant == 2:  # Agast
        impl = cv.AgastFeatureDetector_create(55)
        brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
        print(f'Brief size={brief.descriptorSize()}')

    print(f'Descriptor size={impl.descriptorSize()}')
    print(f'Descriptor type={impl.descriptorType()}')

    return impl.detect(image)


def adaptive_non_maximal_suppression(keypoints, num_to_keep):
    """
    Inspired from: https://answers.opencv.org/question/93317/orb-keypoints-distribution-over-an-image/
    """
    print(f'anms points={len(keypoints)}')
    if len(keypoints) < num_to_keep:
        return keypoints

    sorted_keypoints = sorted(
        keypoints, key=lambda pt: pt.response, reverse=True)
    radii = list()
    radii_sorted = list()

    robust_coeff = 1.11
    for kpt_i in sorted_keypoints:
        response = kpt_i.response * robust_coeff
        radius = sys.float_info.max

        for kpt_j in sorted_keypoints:
            if not kpt_j is kpt_i and kpt_j.response > response:
                pt_i = np.array(kpt_i.pt)
                pt_j = np.array(kpt_j.pt)
                norm = np.linalg.norm(pt_i - pt_j)
                radius = min(radius, norm)
            else:
                break

        radii.append(radius)
        radii_sorted.append(radius)
        if len(radii) % 1000 == 0:
            print(f'Done with iteration={len(radii)}')

    radii_sorted.sort(reverse=True)

    print(radii_sorted[0])
    print(radii_sorted[len(radii_sorted) - 1])

    descision_radius = radii_sorted[num_to_keep]
    anms_points = list()

    for i, radius in enumerate(radii):
        if radius >= descision_radius:
            anms_points.append(sorted_keypoints[i])

    return anms_points


# void adaptiveNonMaximalSuppresion( std::vector<cv::KeyPoint>& keypoints,
#                                        const int numToKeep )
#     {
#       if( keypoints.size() < numToKeep ) { return; }

#       //
#       // Sort by response
#       //
#       std::sort( keypoints.begin(), keypoints.end(),
#                  [&]( const cv::KeyPoint& lhs, const cv::KeyPoint& rhs )
#                  {
#                    return lhs.response > rhs.response;
#                  } );

#       std::vector<cv::KeyPoint> anmsPts;

#       std::vector<double> radii;
#       radii.resize( keypoints.size() );
#       std::vector<double> radiiSorted;
#       radiiSorted.resize( keypoints.size() );

#       const float robustCoeff = 1.11; // see paper

#       for( int i = 0; i < keypoints.size(); ++i )
#       {
#         const float response = keypoints[i].response * robustCoeff;
#         double radius = std::numeric_limits<double>::max();
#         for( int j = 0; j < i && keypoints[j].response > response; ++j )
#         {
#           radius = std::min( radius, cv::norm( keypoints[i].pt - keypoints[j].pt ) );
#         }
#         radii[i]       = radius;
#         radiiSorted[i] = radius;
#       }

#       std::sort( radiiSorted.begin(), radiiSorted.end(),
#                  [&]( const double& lhs, const double& rhs )
#                  {
#                    return lhs > rhs;
#                  } );

#       const double decisionRadius = radiiSorted[numToKeep];
#       for( int i = 0; i < radii.size(); ++i )
#       {
#         if( radii[i] >= decisionRadius )
#         {
#           anmsPts.push_back( keypoints[i] );
#         }
#       }

#       anmsPts.swap( keypoints );
#     }
