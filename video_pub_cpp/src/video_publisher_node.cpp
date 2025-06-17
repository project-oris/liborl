#include <rclcpp/rclcpp.hpp>
#include <image_transport/image_transport.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/msg/image.hpp>

class VideoPublisher : public rclcpp::Node
{
public:
    VideoPublisher()
        : Node("video_publisher")
    {
        // Declare a parameter for the video file path
        this->declare_parameter<std::string>("video_path", "");

        // Get the video file path from the parameter
        std::string video_path = this->get_parameter("video_path").as_string();

        if (video_path.empty())
        {
            RCLCPP_ERROR(this->get_logger(), "No video file path provided. Please set the 'video_path' parameter.");
            rclcpp::shutdown();
            return;
        }

        cap.open(video_path); // Open the video file

        if (!cap.isOpened())
        {
            RCLCPP_ERROR(this->get_logger(), "Error: Could not open video file: %s", video_path.c_str());
            rclcpp::shutdown();
            return;
        }

        this->declare_parameter<std::string>("topic_name", "camera/image_raw");
        std::string topic_name = this->get_parameter("topic_name").as_string();

        publisher_ = image_transport::create_publisher(this, topic_name);

        // Use a timer to publish frames periodically
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(33), // Approximately 30 Hz (1000ms / 30fps)
            std::bind(&VideoPublisher::publish_frame, this));

        RCLCPP_INFO(this->get_logger(), "Publishing video frames from %s", video_path.c_str());
    }

private:
    void publish_frame()
    {
        cv::Mat frame;
        cap >> frame; // Read a new frame from the video

        if (frame.empty())
        {
            RCLCPP_INFO(this->get_logger(), "End of video file. Looping...");
            // Reopen the video to loop
            // Note: For looping, you might want to get the original video path again if it's not stored
            // This example assumes you want to loop the same video
            std::string video_path = this->get_parameter("video_path").as_string();
            cap.open(video_path);
            if (!cap.isOpened())
            {
                RCLCPP_ERROR(this->get_logger(), "Error: Could not reopen video file for looping: %s", video_path.c_str());
                rclcpp::shutdown(); // Shutdown if can't reopen
                return;
            }
            cap >> frame; // Read the first frame of the new loop
            if (frame.empty())
            {
                RCLCPP_ERROR(this->get_logger(), "Error: Frame is still empty after reopening video.");
                rclcpp::shutdown(); // Shutdown if frame is still empty
                return;
            }
        }

        // Convert OpenCV Mat to ROS Image message
        try
        {
            std_msgs::msg::Header header;
            header.stamp = this->now(); // Set the timestamp for the image
            // "bgr8" for color images, "mono8" for grayscale
            sensor_msgs::msg::Image::SharedPtr msg = cv_bridge::CvImage(header, "bgr8", frame).toImageMsg();
            publisher_.publish(*msg);
        }
        catch (cv_bridge::Exception &e)
        {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        }
    }

    cv::VideoCapture cap;
    image_transport::Publisher publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<VideoPublisher>());
    rclcpp::shutdown();
    return 0;
}