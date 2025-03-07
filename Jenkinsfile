pipeline {
    agent any

    environment {
        // Load the Docker Hub credentials stored in Jenkins
        DOCKERHUB_CREDENTIALS = credentials('dockerhub_credentials')
    }

    stages {
        stage('Checkout') {
            steps {
                // Check out the code from the repository
                checkout scm
            }
        }
        stage('Verify Docker') {
            steps {
                sh 'docker --version'
            }
        }
        stage('Build Docker Image') {
            steps {
                script {
                    // Build the Docker image
                    sh "docker build -t asimfatima/flask-svm-app:latest ."
                }
            }
        }
        stage('Push Docker Image') {
            steps {
                script {
                    withCredentials([usernamePassword(credentialsId: 'dockerhub_credentials', passwordVariable: 'DOCKERHUB_PSW', usernameVariable: 'DOCKERHUB_USR')]) {
                        // Log in to Docker Hub securely
                        sh "echo \$DOCKERHUB_PSW | docker login -u \$DOCKERHUB_USR --password-stdin"
                        // Push the Docker image to Docker Hub
                        sh "docker push asimfatima/flask-svm-app:latest"
                    }
                }
            }
        }
    }
    post {
        always {
            // Clean up Docker environment after the job
            sh 'docker rmi asimfatima/flask-svm-app:latest || true'

            // Send an email notification with HTML body
            emailext (
                subject: "Pipeline Status: ${env.BUILD_NUMBER}",
                body: """
                    <html>
                        <body>
                            <p>Build Status: ${currentBuild.currentResult}</p>
                            <p>Build Number: ${env.BUILD_NUMBER}</p>
                            <p>Check the <a href="${env.BUILD_URL}">console output</a> for details.</p>
                        </body>
                    </html>
                """,
                to: 'asimfatima785@gmail.com, i200847@nu.edu.pk, hellojoker9320@gmail.com',
                from: 'jenkins@example.com',
                replyTo: 'jenkins@example.com',
                mimeType: 'text/html'
            )
        }
    }
}
