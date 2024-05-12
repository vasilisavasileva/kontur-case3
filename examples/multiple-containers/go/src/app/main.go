package main

func main() {
	 // Init gin router
	 router := gin.Default()

	 // Its great to version your API's
	 v1 := router.Group("/api/v1")
	 {
		 // Define the hello controller
		 hello := new(controllers.HelloWorldController)
		 // Define a GET request to call the Default
		 // method in controllers/hello.go
		 v1.GET("/hello", hello.Default)
	 }
 
	 // Handle error response when a route is not defined
	 router.NoRoute(func(c *gin.Context) {
		 // In gin this is how you return a JSON response
		 c.JSON(404, gin.H{"message": "Not found"})
	 })

	route.Run(":3004")

}