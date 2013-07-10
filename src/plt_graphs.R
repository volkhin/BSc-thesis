directory <- "../results/"

refine.partition <- function(data) {
  name <- names(data)[1]
  from <- data[1,1]
  to <- data[nrow(data),1]
  result <- data.frame(seq(from, to, length.out=100))  
  names(result) <- name
  result
}

init <- function(name) {
  cairo_pdf(paste0(directory, name, ".pdf"))
  par(cex.lab=1.5, cex.axis=1.5, cex.main=1.5, cex.sub=1.5)
  data <- read.delim(paste0(directory, name, ".txt"))
  data
}

name <- "P2pCollisionTimeOnDiffusionExperiment"
data <- init(name)
data <- data[-c(1:3),]
g <- glm(collision_time ~ cbind(1/diffusion**(1/1)), data=data)
plot.data <- refine.partition(data)
plot(data$diffusion, data$collision_time, xlab=expression("Коэффициент диффузии, мкм"^2~"/с"), ylab="Время между столкновениями частиц, с", xlim=c(0,1e-5), ylim=c(0,50))
lines(plot.data$diffusion, predict(g, plot.data), col="red")
label <- paste("y = ", format(g$coeff[2], scientific=T, digits=4), "/ x")
legend("topright", legend=c(label), inset=0.1)
dev.off()

name <- "P2pCollisionTimeOnConcetrationExperiment"
data <- init(name)
data <- data[-c(1:10),]
g <- glm(collision_time ~ cbind(1/number_of_particles), data=data)
plot.data <- refine.partition(data)
plot(data$number_of_particles, data$collision_time, xlab="Число частиц, шт.", ylab="Время между столкновениями частиц, с", xlim=c(0, 600), ylim=c(0,25))
lines(plot.data$number_of_particles, predict(g, plot.data), col="red")
label <- paste("y = ", format(g$coeff[2], scientific=T, digits=4), " / x")
legend("topright", legend=c(label), inset=0.1)
dev.off()

name <- "P2mCollisionTimeOnLengthExperiment"
data <- init(name)
# data <- data[-c(1:10),]
g <- glm(collision_time ~ cbind(1/microtubule_length**(1/3)), data=data)
plot.data <- refine.partition(data)
plot(data$microtubule_length, data$collision_time, xlab="Длина микротрубочек, мкм", ylab="Время между столкновениями СГ с МТ, с", xlim=c(0, 6), ylim=c(0,140))
lines(plot.data$microtubule_length, predict(g, plot.data), col="red")
label <- paste("y = x^(-1/3) * ", format(g$coeff[2], scientific=F, digits=4))
legend("topright", legend=c(label), inset=0.1)
dev.off()

name <- "P2mCollisionTimeOnRadiusExperiment"
data <- init(name)
# data <- data[-c(1:2),]
g <- glm(collision_time ~ cbind(1/radius**(2/3)), data=data)
plot.data <- refine.partition(data)
plot(data$radius, data$collision_time, xlab="Радиус частиц и микротрубочек, мкм", ylab="Время между столкновениями СГ с МТ, с", xlim=c(0, 0.2), ylim=c(0,150))
lines(plot.data$radius, predict(g, plot.data), col="red")
label <- paste("y = x^(-2/3) * ", format(g$coeff[2], scientific=T, digits=4))
legend("topright", legend=c(label), inset=0.1)
dev.off()

name <- "CreatingLargeGranulesExperiment"
data <- init(name)
# data <- data[-c(1:10),]
g <- glm(time ~ cbind(desired_radius**(1.8)) - 1, data=data)
plot.data <- refine.partition(data)
plot(data$desired_radius, data$time, xlab="Радиус частиц, мкм", ylab="Время образования гранулы, с", xlim=c(0,0.06), ylim=c(0,700))
lines(plot.data$desired_radius, predict(g, plot.data), col="red")
label <- paste("y = x^1.8 * ", format(g$coeff[1], scientific=T, digits=4))
legend("topright", legend=c(label), inset=0.1)
dev.off()

name <- "P2pCollisionTimeOnVolumeExperiment"
data <- init(name)
# data <- data[-c(1:3),]
g <- glm(collision_time ~ cbind(1/board_size**(2)), data=data)
plot.data <- refine.partition(data)
plot(data$board_size, data$collision_time, xlab=expression("Размер боковой стороный среды, мкм"), ylab="Время между столкновениями частиц, с", xlim=c(0,10), ylim=c(0,100))
lines(plot.data$board_size, predict(g, plot.data), col="red")
label <- paste("y = ", format(g$coeff[2], scientific=T, digits=4), " / x")
legend("topright", legend=c(label), inset=0.1)
dev.off()

name <- "P2pCollisionTimeOnImmobilityExperiment"
data <- init(name)
# data <- data[-c(1:3),]
g <- glm(collision_time ~ cbind(immobile_threshold), data=data)
plot.data <- refine.partition(data)
plot(data$immobile_threshold, data$collision_time, xlab=expression("Порог неподвижности, мкм"), ylab="Время между столкновениями частиц, с", xlim=c(0,0.4), ylim=c(0,100))
lines(plot.data$immobile_threshold, predict(g, plot.data), col="red")
# label <- paste("y = 1/x * ", format(g$coeff[2], scientific=T, digits=4))
# legend("topright", legend=c(label), inset=0.1)
dev.off()