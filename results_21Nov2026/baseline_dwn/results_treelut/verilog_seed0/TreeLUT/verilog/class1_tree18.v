module class1_tree18(input wire [467:0] i, output wire [0:0] o);

reg [0:0] o_reg;
assign o = o_reg;

always @(*) begin
	if(((~i[266])) | ((~i[450]) & i[266]) | ((~i[76]) & i[450] & i[266]) | ((~i[47]) & i[106] & i[76] & i[450] & i[266]) | (i[189] & (~i[106]) & i[76] & i[450] & i[266]) | ((~i[189]) & (~i[106]) & i[76] & i[450] & i[266]))
		o_reg = 0;
	else
		o_reg = 1;
end

endmodule
